import csv
import os
import time
from datetime import datetime

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def log(message: str) -> None:
    """Print a timestamped log message."""
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{now} {message}")


class RateLimiter:
    """Simple rate limiter for the Jolpica API."""

    def __init__(self, max_per_sec: int = 4, max_per_hour: int = 500) -> None:
        self.interval = 1.0 / max_per_sec
        self.max_per_hour = max_per_hour
        self.hour_start = time.monotonic()
        self.count = 0
        self.last_request = 0.0

    def wait(self) -> None:
        """Block until a new request is allowed."""
        now = time.monotonic()

        # Reset hourly window if needed
        if now - self.hour_start >= 3600:
            self.hour_start = now
            self.count = 0

        # If the limit has been reached, sleep until the hour resets
        if self.count >= self.max_per_hour:
            time.sleep(3600 - (now - self.hour_start))
            self.hour_start = time.monotonic()
            self.count = 0
            now = self.hour_start

        elapsed = now - self.last_request
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)

        self.last_request = time.monotonic()
        self.count += 1


rate_limiter = RateLimiter()


def fetch_json(url: str):
    """Fetch JSON from the Jolpica F1 API and return the 'MRData' section.

    Retries automatically if a 429 status code is received."""
    retries = 5
    for attempt in range(retries):
        rate_limiter.wait()
        log(f"🔗 GET {url}")
        resp = requests.get(url)
        if resp.status_code == 429:
            delay = 2 ** attempt
            log(f"⏳ 429 received, retrying in {delay}s")
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json().get("MRData", {})

    resp.raise_for_status()


def parse_qual_time(time_str: str):
    """Convert a qualifying lap time 'm:ss.sss' to seconds."""
    if not time_str:
        return None
    try:
        minutes, sec = time_str.split(":")
        return int(minutes) * 60 + float(sec)
    except ValueError:
        return None


def get_qualifying_results(season: int, round_no: int):
    """Return the list of qualifying results for a race."""
    url = f"{BASE_URL}/{season}/{round_no}/qualifying.json"
    data = fetch_json(url)
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0].get("QualifyingResults", [])
    return []


def get_results(season: int, round_no: int):
    """Return circuit id and race results for a given season and round."""
    url = f"{BASE_URL}/{season}/{round_no}/results.json"
    data = fetch_json(url)
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        race = races[0]
        circuit_id = race.get("Circuit", {}).get("circuitId")
        return circuit_id, race.get("Results", [])
    return None, []


def get_driver_standings(season: int, round_no: int):
    """Return driver standings after a given round."""
    url = f"{BASE_URL}/{season}/{round_no}/driverStandings.json"
    data = fetch_json(url)
    lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    if lists:
        return lists[0].get("DriverStandings", [])
    return []


def get_constructor_standings(season: int, round_no: int):
    """Return constructor standings after a given round."""
    url = f"{BASE_URL}/{season}/{round_no}/constructorStandings.json"
    data = fetch_json(url)
    lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    if lists:
        return lists[0].get("ConstructorStandings", [])
    return []


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


def collect_data(start_season: int, end_season: int, output_file: str):
    """Collect race level data from start_season to end_season (inclusive).

    If ``output_file`` already exists, the script resumes from the last
    recorded race and appends new rows."""

    log(f"🏁 Collecting data from {start_season} to {end_season}")
    last = get_last_round(output_file)
    mode = "a" if last else "w"

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
                "q2_flag",
                "q3_flag",
                "driver_points_scored",
                "driver_championship_rank",
                "constructor_id",
                "constructor_points_scored",
                "constructor_championship_rank",
                "rqtd_sec",
                "rqtd_pct",
            ])

        if last:
            log(f"↩️ Resuming from {last[0]} round {last[1]}")

        start_s = last[0] if last else start_season
        start_r = last[1] + 1 if last else 1

        for season in range(start_s, end_season + 1):
            round_no = start_r if season == start_s else 1
            while True:
                log(f"🚦 {season} round {round_no}")
                circuit_id, results = get_results(season, round_no)
                if not results:
                    break

                driver_standings = get_driver_standings(season, round_no)
                cons_standings = get_constructor_standings(season, round_no)
                qual_results = get_qualifying_results(season, round_no)

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

                    try:
                        grid_pos = int(result.get("grid"))
                    except (TypeError, ValueError):
                        grid_pos = None
                    qual_pos = qual_positions.get(driver)
                    if grid_pos is not None and qual_pos is not None:
                        penalty_places = grid_pos - qual_pos
                    else:
                        penalty_places = None
                    penalty_flag = 1 if penalty_places is not None and penalty_places > 0 else 0

                    writer.writerow([
                        season,
                        round_no,
                        circuit_id,
                        driver,
                        result.get("grid"),
                        result.get("position"),
                        penalty_places,
                        penalty_flag,
                        qual_flags.get(driver, (0, 0))[0],
                        qual_flags.get(driver, (0, 0))[1],
                        ds.get("points"),
                        ds.get("position"),
                        constructor,
                        cs.get("points"),
                        cs.get("position"),
                        best_times.get(driver) - pole_time if best_times.get(driver) is not None and pole_time is not None else None,
                        (best_times.get(driver) / pole_time - 1) * 100 if best_times.get(driver) is not None and pole_time is not None else None,
                    ])

                log(f"✅ stored {len(results)} results for {season} round {round_no}")
                round_no += 1

            start_r = 1


if __name__ == "__main__":
    current_year = datetime.now().year
    collect_data(2022, current_year, "f1_data_2022_to_present.csv")
