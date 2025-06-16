"""Prepare a CSV dataset from cached Jolpica F1 race data."""

import csv
import json
import os
from datetime import datetime

from fetch_data import fetch_round_data, log

# window size for computing DNF rates
DNF_WINDOW = 5


def is_dnf(status: str) -> bool:
    """Return True if the given status indicates a retirement."""
    if not status:
        return True
    status = status.lower()
    return not ("finished" in status or "lap" in status)


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



def load_weather(season: int, round_no: int):
    """Load cached weather forecast features for a race."""
    cache_file = os.path.join("weather_cache", f"weather_{season}_{round_no}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)
    # If the file is missing, attempt to fetch and cache it
    try:
        from fetch_data import fetch_weather

        return fetch_weather(season, round_no)
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
                "driver_last3_performance",
                "driver_momentum",
                "constructor_last3_performance",
                "constructor_momentum",
                "circuit_podium_rate",
                "constructor_podium_rate",
                "driver_dnf_rate",
                "constructor_dnf_rate",
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
        driver_dnf_history = {}
        constructor_dnf_history = {}
        circuit_pit_history = {}

        for season in range(start_s, end_season + 1):
            round_no = start_r if season == start_s else 1
            while True:
                log(f"🚦 {season} round {round_no}")
                data = fetch_round_data(season, round_no)
                if data is None:
                    break
                circuit_id = data["circuit_id"]
                results = data["results"]
                driver_standings_curr = data["driver_standings"]
                cons_standings_curr = data["constructor_standings"]
                qual_results = data["qualifying"]

                if round_no == 1:
                    driver_standings_prev = []
                    cons_standings_prev = []
                else:
                    prev = fetch_round_data(season, round_no - 1)
                    driver_standings_prev = prev.get("driver_standings", [])
                    cons_standings_prev = prev.get("constructor_standings", [])

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

                current_psd = None
                if pit_durations:
                    avg_dur = sum(pit_durations) / len(pit_durations)
                    current_psd = len(pit_durations) * avg_dur

                past_diffs = circuit_pit_history.get(circuit_id, [])
                pit_stop_difficulty = (
                    sum(past_diffs) / len(past_diffs) if past_diffs else None
                )

                weather = load_weather(season, round_no)

                # Convert standings to dicts for quick lookup
                ds_prev_map = {d["Driver"]["driverId"]: d for d in driver_standings_prev}
                cs_prev_map = {
                    c["Constructor"]["constructorId"]: c
                    for c in cons_standings_prev
                }

                ds_curr_map = {d["Driver"]["driverId"]: d for d in driver_standings_curr}
                cs_curr_map = {
                    c["Constructor"]["constructorId"]: c
                    for c in cons_standings_curr
                }

                for result in results:
                    driver = result["Driver"]["driverId"]
                    constructor = result["Constructor"]["constructorId"]
                    ds = ds_prev_map.get(driver, {})
                    cs = cs_prev_map.get(constructor, {})

                    # Championship rank with fallback to last known value
                    rank = try_int(ds.get("position"))
                    if rank is None:
                        rank = last_driver_rank.get(driver)
                    post_rank = try_int(ds_curr_map.get(driver, {}).get("position"))
                    if post_rank is not None:
                        last_driver_rank[driver] = post_rank

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

                    points_total_prev = try_float(ds.get("points"))
                    history = points_history.setdefault(driver, [])

                    # average points scored in the previous 3 races (leakage-safe)
                    if len(history) >= 4:
                        last3_perf = (history[-1] - history[-4]) / 3
                    elif history:
                        last3_perf = (history[-1] - history[0]) / len(history)
                    else:
                        last3_perf = 0.0

                    points_after = try_float(ds_curr_map.get(driver, {}).get("points"))

                    momentum = None
                    if len(history) >= 6:
                        last3 = history[-1] - history[-4]
                        prev3 = history[-4] - history[-6]
                        momentum = last3 - prev3
                    else:
                        momentum = 0.0

                    history.append(
                        points_after if points_after is not None else (history[-1] if history else 0.0)
                    )

                    cons_points = try_float(cs.get("points"))
                    cons_hist = constructor_points_history.setdefault(constructor, [])
                    cons_points_after = try_float(cs_curr_map.get(constructor, {}).get("points"))

                    if len(cons_hist) >= 4:
                        cons_last3_perf = (cons_hist[-1] - cons_hist[-4]) / 3
                    elif cons_hist:
                        cons_last3_perf = (cons_hist[-1] - cons_hist[0]) / len(cons_hist)
                    else:
                        cons_last3_perf = 0.0

                    cons_momentum = None
                    if len(cons_hist) >= 6:
                        last3_c = cons_hist[-1] - cons_hist[-4]
                        prev3_c = cons_hist[-4] - cons_hist[-6]
                        cons_momentum = last3_c - prev3_c
                    else:
                        cons_momentum = 0.0

                    cons_hist.append(
                        cons_points_after
                        if cons_points_after is not None
                        else (cons_hist[-1] if cons_hist else 0.0)
                    )

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

                    circuit_podium_rate = (
                        circ_pods / circ_count if circ_count else 0.0
                    )
                    constructor_podium_rate = (
                        cons_pods / cons_count if cons_count else 0.0
                    )

                    d_hist = driver_dnf_history.get(driver, [])
                    c_hist = constructor_dnf_history.get(constructor, [])
                    driver_dnf_rate = (
                        sum(d_hist[-DNF_WINDOW:]) / len(d_hist[-DNF_WINDOW:])
                        if d_hist
                        else 0.0
                    )
                    constructor_dnf_rate = (
                        sum(c_hist[-DNF_WINDOW:]) / len(c_hist[-DNF_WINDOW:])
                        if c_hist
                        else 0.0
                    )

                    dnf_flag = 1 if is_dnf(result.get("status")) else 0

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
                        last3_perf,
                        momentum,
                        cons_last3_perf,
                        cons_momentum,
                        circuit_podium_rate,
                        constructor_podium_rate,
                        driver_dnf_rate,
                        constructor_dnf_rate,
                        pit_stop_difficulty,
                        weather.get("temp_mean"),
                        weather.get("precip_sum"),
                        weather.get("humidity_mean"),
                        weather.get("wind_mean"),
                    ])

                    # Update statistics after writing row
                    driver_dnf_history.setdefault(driver, []).append(dnf_flag)
                    constructor_dnf_history.setdefault(constructor, []).append(
                        dnf_flag
                    )
                    if finish_pos is not None:
                        circuit_counts[circuit_id] = circ_count + 1
                        constructor_counts[constructor] = cons_count + 1
                        if finish_pos <= 3:
                            circuit_podiums[circuit_id] = circ_pods + 1
                            constructor_podiums[constructor] = cons_pods + 1

                if current_psd is not None:
                    circuit_pit_history.setdefault(circuit_id, []).append(current_psd)

                log(f"✅ stored {len(results)} results for {season} round {round_no}")
                round_no += 1

            start_r = 1


if __name__ == "__main__":
    current_year = datetime.now().year
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
