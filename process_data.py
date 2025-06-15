"""Prepare a CSV dataset from cached Jolpica F1 race data."""

import csv
import json
import os
from datetime import datetime

from meteostat import Hourly, Point
from pyowm.owm import OWM

from fetch_data import fetch_round_data, log


def parse_qual_time(time_str: str):
    """Convert a qualifying lap time 'm:ss.sss' or 'ss.sss' to seconds."""
    if not time_str:
        return None
    if ":" not in time_str:
        # Some qualifying times are reported without a minutes component
        try:
            return float(time_str)
        except ValueError:
            return None
    try:
        minutes, sec = time_str.split(":")
        return int(minutes) * 60 + float(sec)
    except ValueError:
        return None


def try_int(value):
    """Return int(value) or None if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def try_float(value):
    """Return float(value) or None if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_pit_duration(value: str):
    """Convert a pit stop duration string to seconds."""
    if not value:
        return None
    if ":" in value:
        try:
            minutes, sec = value.split(":")
            return int(minutes) * 60 + float(sec)
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def get_last_round(csv_file: str):
    """Return the last processed (season, round) from an existing CSV file."""
    if not os.path.exists(csv_file):
        return None

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        last = None
        for row in reader:
            if len(row) >= 2:
                last = row
        if last:
            try:
                return int(last[0]), int(last[1])
            except ValueError:
                return None
    return None


def fetch_weather(season: int, round_no: int):
    """Return weather features for a race using caching."""
    os.makedirs("weather_cache", exist_ok=True)
    cache_file = os.path.join("weather_cache", f"weather_{season}_{round_no}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    try:
        race = fetch_round_data(season, round_no)
        if race is None:
            return {}
        info = race
        loc = info.get("Circuit", {}).get("Location", {})
        lat = float(loc.get("lat"))
        lon = float(loc.get("long"))
        date_str = info.get("date")
        race_day = datetime.fromisoformat(date_str)

        start = datetime(race_day.year, race_day.month, race_day.day, 12)
        end = datetime(race_day.year, race_day.month, race_day.day, 14)

        now = datetime.utcnow()
        features = {
            "temp_mean": None,
            "precip_sum": None,
            "humidity_mean": None,
            "wind_mean": None,
        }

        if start <= now:
            # historical data via Meteostat
            location = Point(lat, lon)
            data = Hourly(location, start, end)
            df_w = data.fetch()
            if not df_w.empty:
                features = {
                    "temp_mean": float(df_w["temp"].mean()),
                    "precip_sum": float(df_w["prcp"].sum()),
                    "humidity_mean": float(df_w["rhum"].mean()),
                    "wind_mean": float(df_w["wspd"].mean()),
                }
        else:
            api_key = os.getenv("OWM_API_KEY")
            if api_key:
                owm = OWM(api_key)
                mgr = owm.weather_manager()
                fc = mgr.forecast_hourly(lat=lat, lon=lon)
                hours = [
                    h
                    for h in fc.forecast
                    if start <= h.reference_time("date") <= end
                ]
                if hours:
                    temps = [h.temperature("celsius").get("temp") for h in hours]
                    prcps = [h.rain.get("1h", 0.0) for h in hours]
                    hums = [h.humidity for h in hours]
                    winds = [h.wind().get("speed", 0.0) for h in hours]
                    features = {
                        "temp_mean": sum(temps) / len(temps) if temps else None,
                        "precip_sum": sum(prcps) if prcps else None,
                        "humidity_mean": sum(hums) / len(hums) if hums else None,
                        "wind_mean": sum(winds) / len(winds) if winds else None,
                    }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(features, f)
        return features
    except Exception:
        return {}


def prepare_dataset(start_season: int, end_season: int, output_file: str):
    """Prepare CSV data for the given seasons using cached raw data."""

    log(f"📄 Preparing dataset from {start_season} to {end_season}")
    last = get_last_round(output_file)
    mode = "a" if last else "w"

    # Keep track of last known driver standings position
    last_driver_rank = {}

    # Statistics for target/mean encoding of circuits and constructors
    circuit_counts = {}
    circuit_podiums = {}
    constructor_counts = {}
    constructor_podiums = {}

    with open(output_file, mode, newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not last:
            writer.writerow([
                "season_year",
                "round_number",
                "circuit_id",
                "driver_id",
                "starting_grid_position",
                "finishing_position",
                "grid_penalty_places",
                "grid_penalty_flag",
                "grid_bonus_flag",
                "q2_flag",
                "q3_flag",
                "driver_points_scored",
                "driver_championship_rank",
                "constructor_id",
                "constructor_points_scored",
                "constructor_championship_rank",
                "rqtd_sec",
                "rqtd_pct",
                "teammate_quali_gap_sec",
                "driver_momentum",
                "constructor_momentum",
                "pit_stop_difficulty",
                "temp_mean",
                "precip_sum",
                "humidity_mean",
                "wind_mean",
            ])

        if last:
            log(f"↩️ Resuming from {last[0]} round {last[1]}")

        start_s = last[0] if last else start_season
        start_r = last[1] + 1 if last else 1

        points_history = {}
        constructor_points_history = {}

        for season in range(start_s, end_season + 1):
            round_no = start_r if season == start_s else 1
            while True:
                log(f"🚦 {season} round {round_no}")
                data = fetch_round_data(season, round_no)
                if data is None:
                    break
                circuit_id = data["circuit_id"]
                results = data["results"]
                driver_standings = data["driver_standings"]
                cons_standings = data["constructor_standings"]
                qual_results = data["qualifying"]

                # Map best qualifying times in seconds
                best_times = {}
                qual_flags = {}
                qual_positions = {}
                for qr in qual_results:
                    drv = qr["Driver"]["driverId"]
                    t1 = parse_qual_time(qr.get("Q1"))
                    t2 = parse_qual_time(qr.get("Q2"))
                    t3 = parse_qual_time(qr.get("Q3"))
                    times = [t for t in (t1, t2, t3) if t is not None]
                    best_times[drv] = min(times) if times else None
                    try:
                        pos = int(qr.get("position"))
                    except (TypeError, ValueError):
                        pos = None
                    q2_flag = 1 if pos is not None and pos <= 15 else 0
                    q3_flag = 1 if pos is not None and pos <= 10 else 0
                    qual_flags[drv] = (q2_flag, q3_flag)
                    if pos is not None:
                        qual_positions[drv] = pos

                pole_time = None
                if best_times:
                    vals = [t for t in best_times.values() if t is not None]
                    if vals:
                        pole_time = min(vals)

                team_map = {}
                for res in results:
                    drv_id = res["Driver"]["driverId"]
                    team_id = res["Constructor"]["constructorId"]
                    team_map.setdefault(team_id, []).append(drv_id)

                pit_durations = []
                for p in data.get("pitstops", []):
                    dur = parse_pit_duration(p.get("duration"))
                    if dur is not None:
                        pit_durations.append(dur)
                pit_stop_difficulty = None
                if pit_durations:
                    avg_dur = sum(pit_durations) / len(pit_durations)
                    pit_stop_difficulty = len(pit_durations) * avg_dur

                weather = fetch_weather(season, round_no)

                # Convert standings to dicts for quick lookup
                ds_map = {d["Driver"]["driverId"]: d for d in driver_standings}
                cs_map = {c["Constructor"]["constructorId"]: c for c in cons_standings}

                for result in results:
                    driver = result["Driver"]["driverId"]
                    constructor = result["Constructor"]["constructorId"]
                    ds = ds_map.get(driver, {})
                    cs = cs_map.get(constructor, {})

                    # Championship rank with fallback to last known value
                    rank = try_int(ds.get("position"))
                    if rank is None:
                        rank = last_driver_rank.get(driver)
                    else:
                        last_driver_rank[driver] = rank

                    # Historical counts for target/mean encoding
                    circ_count = circuit_counts.get(circuit_id, 0)
                    circ_pods = circuit_podiums.get(circuit_id, 0)
                    cons_count = constructor_counts.get(constructor, 0)
                    cons_pods = constructor_podiums.get(constructor, 0)

                    grid_pos = try_int(result.get("grid"))
                    finish_pos = try_int(result.get("position"))
                    qual_pos = qual_positions.get(driver)
                    if grid_pos is not None and qual_pos is not None:
                        penalty_places = grid_pos - qual_pos
                    else:
                        penalty_places = None
                    penalty_flag = 1 if penalty_places is not None and penalty_places > 0 else 0
                    bonus_flag = 1 if penalty_places is not None and penalty_places < 0 else 0

                    teammates = [t for t in team_map.get(constructor, []) if t != driver]
                    teammate_best = None
                    if teammates:
                        times = [best_times.get(t) for t in teammates if best_times.get(t) is not None]
                        if times:
                            teammate_best = min(times)
                    teammate_gap = (
                        best_times.get(driver) - teammate_best
                        if best_times.get(driver) is not None and teammate_best is not None
                        else None
                    )
                    if teammate_gap is None:
                        teammate_gap = 5.0

                    points_total = try_float(ds.get("points"))
                    history = points_history.setdefault(driver, [])
                    history.append(points_total if points_total is not None else 0.0)
                    momentum = None
                    if len(history) >= 7:
                        last3 = history[-1] - history[-4]
                        prev3 = history[-4] - history[-7]
                        momentum = last3 - prev3
                    else:
                        momentum = 0.0

                    cons_points = try_float(cs.get("points"))
                    cons_hist = constructor_points_history.setdefault(constructor, [])
                    cons_hist.append(cons_points if cons_points is not None else 0.0)
                    cons_momentum = None
                    if len(cons_hist) >= 7:
                        last3_c = cons_hist[-1] - cons_hist[-4]
                        prev3_c = cons_hist[-4] - cons_hist[-7]
                        cons_momentum = last3_c - prev3_c
                    else:
                        cons_momentum = 0.0

                    gap_sec = (
                        best_times.get(driver) - pole_time
                        if best_times.get(driver) is not None and pole_time is not None
                        else None
                    )
                    gap_pct = (
                        (best_times.get(driver) / pole_time - 1) * 100
                        if best_times.get(driver) is not None and pole_time is not None
                        else None
                    )
                    gap_sec = gap_sec if gap_sec is not None else 5.0
                    gap_pct = gap_pct if gap_pct is not None else 5.0

                    writer.writerow([
                        season,
                        round_no,
                        circuit_id,
                        driver,
                        grid_pos,
                        finish_pos,
                        penalty_places,
                        penalty_flag,
                        bonus_flag,
                        qual_flags.get(driver, (0, 0))[0],
                        qual_flags.get(driver, (0, 0))[1],
                        try_float(ds.get("points")),
                        rank,
                        constructor,
                        try_float(cs.get("points")),
                        try_int(cs.get("position")),
                        gap_sec,
                        gap_pct,
                        teammate_gap,
                        momentum,
                        cons_momentum,
                        pit_stop_difficulty,
                        weather.get("temp_mean"),
                        weather.get("precip_sum"),
                        weather.get("humidity_mean"),
                        weather.get("wind_mean"),
                    ])

                    # Update statistics after writing row
                    if finish_pos is not None:
                        circuit_counts[circuit_id] = circ_count + 1
                        constructor_counts[constructor] = cons_count + 1
                        if finish_pos <= 3:
                            circuit_podiums[circuit_id] = circ_pods + 1
                            constructor_podiums[constructor] = cons_pods + 1

                log(f"✅ stored {len(results)} results for {season} round {round_no}")
                round_no += 1

            start_r = 1


if __name__ == "__main__":
    current_year = datetime.now().year
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
