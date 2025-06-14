# === process_data.py (updated) ===
"""Prepare a CSV dataset from cached Jolpica F1 race data, including
calculated overtake counts per driver.
"""

import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from fetch_data import fetch_round_data, log

# ---------------- Helper conversions ----------------

def parse_qual_time(time_str: str):
    if not time_str:
        return None
    if ":" not in time_str:
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
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def try_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_pit_duration(value: str):
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

# ---------------- NEW: Overtake logic ----------------

def tally_overtakes(laps: List[Dict]):
    """Return per‑driver dicts of overtakes made and lost.

    We skip the first lap (formation + start chaos) and count a pass only
    when a driver exits a lap in a better position than on the previous lap.
    """
    pos_prev: Dict[str, int] = {}
    made: Dict[str, int] = defaultdict(int)
    lost: Dict[str, int] = defaultdict(int)

    if not laps or len(laps) < 2:
        return made, lost

    for lap in laps[1:]:  # start at second flying lap
        for timing in lap.get("Timings", []):
            drv = timing.get("driverId")
            pos = try_int(timing.get("position"))
            if drv is None or pos is None:
                continue
            if drv in pos_prev:
                delta = pos_prev[drv] - pos
                if delta > 0:
                    made[drv] += delta
                elif delta < 0:
                    lost[drv] += -delta
            pos_prev[drv] = pos

    return made, lost

# ---------------- Core routine ----------------

def get_last_round(csv_file: str):
    if not os.path.exists(csv_file):
        return None
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        last = None
        for row in reader:
            if row:
                last = row
        if last and len(last) >= 2:
            try:
                return int(last[0]), int(last[1])
            except ValueError:
                return None
    return None


def prepare_dataset(start_season: int, end_season: int, output_file: str):
    log(f"📄 Preparing dataset from {start_season} to {end_season}")
    last = get_last_round(output_file)
    mode = "a" if last else "w"

    last_driver_rank: Dict[str, int] = {}
    circuit_counts: Dict[str, int] = {}
    circuit_podiums: Dict[str, int] = {}
    constructor_counts: Dict[str, int] = {}
    constructor_podiums: Dict[str, int] = {}

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
                # 👉 new fields
                "overtakes_made",
                "overtakes_lost",
                "net_overtakes",
            ])

        start_s = last[0] if last else start_season
        start_r = last[1] + 1 if last else 1

        points_history: Dict[str, List[float]] = {}
        constructor_points_history: Dict[str, List[float]] = {}

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
                laps = data.get("laps", [])

                # Map best qualifying times
                best_times, qual_flags, qual_positions = {}, {}, {}
                for qr in qual_results:
                    drv = qr["Driver"]["driverId"]
                    t_vals = [parse_qual_time(qr.get(q)) for q in ("Q1", "Q2", "Q3")]
                    valid = [t for t in t_vals if t is not None]
                    best_times[drv] = min(valid) if valid else None
                    pos = try_int(qr.get("position"))
                    q2_flag = 1 if pos is not None and pos <= 15 else 0
                    q3_flag = 1 if pos is not None and pos <= 10 else 0
                    qual_flags[drv] = (q2_flag, q3_flag)
                    if pos is not None:
                        qual_positions[drv] = pos

                pole_time = min([t for t in best_times.values() if t is not None], default=None)

                # team mapping
                team_map: Dict[str, List[str]] = {}
                for res in results:
                    d_id = res["Driver"]["driverId"]
                    c_id = res["Constructor"]["constructorId"]
                    team_map.setdefault(c_id, []).append(d_id)

                # pit stop difficulty (unchanged)
                pit_durations = [parse_pit_duration(p.get("duration")) for p in data.get("pitstops", [])]
                pit_durations = [d for d in pit_durations if d is not None]
                pit_stop_difficulty = None
                if pit_durations:
                    pit_stop_difficulty = len(pit_durations) * (sum(pit_durations) / len(pit_durations))

                # standings lookup
                ds_map = {d["Driver"]["driverId"]: d for d in driver_standings}
                cs_map = {c["Constructor"]["constructorId"]: c for c in cons_standings}

                # 👉 compute overtakes per driver for this race
                made_dict, lost_dict = tally_overtakes(laps)

                for res in results:
                    driver = res["Driver"]["driverId"]
                    constructor = res["Constructor"]["constructorId"]
                    ds = ds_map.get(driver, {})
                    cs = cs_map.get(constructor, {})

                    rank = try_int(ds.get("position"))
                    if rank is None:
                        rank = last_driver_rank.get(driver)
                    else:
                        last_driver_rank[driver] = rank

                    circ_count = circuit_counts.get(circuit_id, 0)
                    circ_pods = circuit_podiums.get(circuit_id, 0)
                    cons_count = constructor_counts.get(constructor, 0)
                    cons_pods = constructor_podiums.get(constructor, 0)

                    grid_pos = try_int(res.get("grid"))
                    finish_pos = try_int(res.get("position"))
                    qual_pos = qual_positions.get(driver)
                    penalty_places = (grid_pos - qual_pos) if (grid_pos is not None and qual_pos is not None) else None
                    penalty_flag = 1 if penalty_places and penalty_places > 0 else 0
                    bonus_flag = 1 if penalty_places and penalty_places < 0 else 0

                    teammates = [t for t in team_map.get(constructor, []) if t != driver]
                    teammate_best = min([best_times.get(t) for t in teammates if best_times.get(t) is not None], default=None)
                    teammate_gap = (best_times.get(driver) - teammate_best) if (best_times.get(driver) and teammate_best) else 5.0

                    points_total = try_float(ds.get("points"))
                    hist = points_history.setdefault(driver, [])
                    hist.append(points_total or 0.0)
                    momentum = 0.0
                    if len(hist) >= 7:
                        momentum = (hist[-1] - hist[-4]) - (hist[-4] - hist[-7])

                    cons_points = try_float(cs.get("points"))
                    chist = constructor_points_history.setdefault(constructor, [])
                    chist.append(cons_points or 0.0)
                    cons_momentum = 0.0
                    if len(chist) >= 7:
                        cons_momentum = (chist[-1] - chist[-4]) - (chist[-4] - chist[-7])

                    gap_sec = (best_times.get(driver) - pole_time) if (best_times.get(driver) and pole_time) else 5.0
                    gap_pct = ((best_times.get(driver) / pole_time - 1) * 100) if (best_times.get(driver) and pole_time) else 5.0

                    # new overtake values (default 0)
                    ov_made = made_dict.get(driver, 0)
                    ov_lost = lost_dict.get(driver, 0)
                    net_ov = ov_made - ov_lost

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
                        ov_made,
                        ov_lost,
                        net_ov,
                    ])

                    # update simple counts
                    if finish_pos is not None:
                        circuit_counts[circuit_id] = circ_count + 1
                        constructor_counts[constructor] = cons_count + 1
                        if finish_pos <= 3:
                            circuit_podiums[circuit_id] = circ_pods + 1
                            constructor_podiums[constructor] = cons_pods + 1

                log(f"✅ stored {len(results)} results for {season} round {round_no}")
                round_no += 1
            start_r = 1  # reset after first season loop


if __name__ == "__main__":
    current_year = datetime.now().year
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
