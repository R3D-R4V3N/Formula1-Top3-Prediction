"""Prepare a CSV dataset from cached Jolpica F1 race data."""

import csv
import os
from datetime import datetime

from fetch_data import fetch_round_data, log


def parse_qual_time(time_str: str):
    """Convert a qualifying lap time 'm:ss.sss' to seconds."""
    if not time_str:
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
                "circuit_podium_rate",
                "constructor_podium_rate",
            ])

        if last:
            log(f"↩️ Resuming from {last[0]} round {last[1]}")

        start_s = last[0] if last else start_season
        start_r = last[1] + 1 if last else 1

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

                # Convert standings to dicts for quick lookup
                ds_map = {d["Driver"]["driverId"]: d for d in driver_standings}
                cs_map = {c["Constructor"]["constructorId"]: c for c in cons_standings}

                for result in results:
                    driver = result["Driver"]["driverId"]
                    constructor = result["Constructor"]["constructorId"]
                    ds = ds_map.get(driver, {})
                    cs = cs_map.get(constructor, {})

                    grid_pos = try_int(result.get("grid"))
                    finish_pos = try_int(result.get("position"))
                    qual_pos = qual_positions.get(driver)
                    if grid_pos is not None and qual_pos is not None:
                        penalty_places = grid_pos - qual_pos
                    else:
                        penalty_places = None
                    penalty_flag = 1 if penalty_places is not None and penalty_places > 0 else 0
                    bonus_flag = 1 if penalty_places is not None and penalty_places < 0 else 0

                    # Handle missing championship rank
                    rank = try_int(ds.get("position"))
                    if rank is None:
                        rank = last_driver_rank.get(driver, 999)
                    else:
                        last_driver_rank[driver] = rank

                    # Sentinel values for missing qualifying times
                    if best_times.get(driver) is not None and pole_time is not None:
                        rqtd_sec = best_times[driver] - pole_time
                        rqtd_pct = (best_times[driver] / pole_time - 1) * 100
                    else:
                        rqtd_sec = 5.0
                        rqtd_pct = 5.0

                    # Target/mean encoding features
                    circ_count = circuit_counts.get(circuit_id, 0)
                    circ_pods = circuit_podiums.get(circuit_id, 0)
                    circuit_rate = circ_pods / circ_count if circ_count > 0 else 0.0

                    cons_count = constructor_counts.get(constructor, 0)
                    cons_pods = constructor_podiums.get(constructor, 0)
                    constructor_rate = cons_pods / cons_count if cons_count > 0 else 0.0

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
                        rqtd_sec,
                        rqtd_pct,
                        circuit_rate,
                        constructor_rate,
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
