"""Prepare a CSV dataset from cached Jolpica F1 race data."""

import csv
import os
from datetime import datetime

import pandas as pd
from pandas import json_normalize

from fetch_data import (
    fetch_round_data,
    get_laps,
    get_pitstops,
    get_status,
    log,
)


# Cache of passes per race for computing overtake difficulty
race_pass_df = pd.DataFrame(
    columns=["season", "round", "circuit_id", "passes"]
)


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


def detect_passes(laps: list, pit_df: list, status_df: list) -> int:
    """Detect genuine on-track overtakes for a race."""
    lap_df = json_normalize(
        laps,
        record_path="Timings",
        meta="number",
    ).rename(
        columns={"number": "lap", "driverId": "driver", "position": "pos"}
    )
    if lap_df.empty:
        return 0

    lap_df["lap"] = lap_df["lap"].astype(int)
    lap_df["pos"] = lap_df["pos"].astype(int)
    lap_df.sort_values(["driver", "lap"], inplace=True)

    lap_df["prev_pos"] = lap_df.groupby("driver")["pos"].shift(1)
    lap_df["gain"] = lap_df["prev_pos"] - lap_df["pos"]

    pits = pd.DataFrame(pit_df)
    pits_by_lap = (
        pits.groupby(pits["lap"].astype(int))["driverId"].apply(set).to_dict()
        if not pits.empty
        else {}
    )

    dnf_drivers = {
        s.get("driverId")
        for s in status_df
        if s.get("status") != "Finished"
    }

    passes = []
    for _, row in lap_df.iterrows():
        if row["lap"] <= 1:
            continue
        if row["driver"] in dnf_drivers:
            continue
        if row["gain"] <= 0:
            continue
        if row["driver"] in pits_by_lap.get(row["lap"], set()):
            continue
        prev_driver = lap_df[
            (lap_df["lap"] == row["lap"] - 1)
            & (lap_df["pos"] == row["prev_pos"])
        ]["driver"]
        if (
            not prev_driver.empty
            and prev_driver.iloc[0] in pits_by_lap.get(row["lap"], set())
        ):
            continue
        passes.append(row["gain"])

    return int(sum(passes))


def circuit_difficulty(df: pd.DataFrame) -> dict:
    """Return mapping of circuit id to overtake difficulty."""
    if df.empty:
        return {}

    max_passes = df["passes"].max()
    difficulty = {}
    for circ, grp in df.groupby("circuit_id"):
        mean_p = grp["passes"].mean()
        difficulty[circ] = 1 - (mean_p / max_passes if max_passes else 0)
    return difficulty


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
                "overtake_difficulty",
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

                diff_map = circuit_difficulty(race_pass_df)
                fallback_diff = 1.0
                overtake_diff = diff_map.get(circuit_id, fallback_diff)

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
                        overtake_diff,
                    ])

                    # Update statistics after writing row
                    if finish_pos is not None:
                        circuit_counts[circuit_id] = circ_count + 1
                        constructor_counts[constructor] = cons_count + 1
                        if finish_pos <= 3:
                            circuit_podiums[circuit_id] = circ_pods + 1
                            constructor_podiums[constructor] = cons_pods + 1

                laps = get_laps(season, round_no)
                pits = get_pitstops(season, round_no)
                statuses = get_status(season, round_no)
                passes = detect_passes(laps, pits, statuses)
                race_pass_df.loc[len(race_pass_df)] = [
                    season,
                    round_no,
                    circuit_id,
                    passes,
                ]

                log(
                    f"✅ stored {len(results)} results for {season} round {round_no}"
                )
                round_no += 1

            start_r = 1


if __name__ == "__main__":
    current_year = datetime.now().year
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
