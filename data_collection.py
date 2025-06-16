from datetime import datetime
import csv
import os
from typing import List, Dict

from fetch_data import (
    BASE_URL,
    fetch_data,
    fetch_json,
    fetch_round_data,
)
from process_data import prepare_dataset


def _fetch_all_laps(season: int, round_no: int) -> List[Dict]:
    """Return the full lap list for a race."""
    laps = []
    offset = 0
    limit = 1000
    while True:
        url = f"{BASE_URL}/{season}/{round_no}/laps.json?limit={limit}&offset={offset}"
        data = fetch_json(url)
        race_table = data.get("RaceTable", {})
        races = race_table.get("Races", [])
        if races:
            laps.extend(races[0].get("Laps", []))
        total = int(data.get("total", 0))
        offset += limit
        if offset >= total:
            break
    return laps


def get_ontrack_passes(season: int, round_no: int, csv_file: str = "overtakes.csv") -> None:
    """Compute net on-track overtakes and append to overtakes.csv."""
    race = fetch_round_data(season, round_no)
    if race is None:
        return

    circuit = race.get("circuit_id")
    pit_laps = {
        int(p.get("lap"))
        for p in race.get("pitstops", [])
        if p.get("lap") is not None
    }
    laps = _fetch_all_laps(season, round_no)
    positions: Dict[str, int] = {}
    passes = 0
    for lap in laps:
        lap_no = int(lap.get("number", 0))
        timings = lap.get("Timings", [])
        if lap_no in pit_laps:
            for t in timings:
                try:
                    positions[t["driverId"]] = int(t["position"])
                except (KeyError, ValueError, TypeError):
                    continue
            continue
        for t in timings:
            try:
                drv = t["driverId"]
                pos = int(t["position"])
            except (KeyError, ValueError, TypeError):
                continue
            prev_pos = positions.get(drv)
            if prev_pos is not None and pos < prev_pos:
                passes += 1
            positions[drv] = pos

    laps_completed = max(
        int(r.get("laps", 0)) for r in race.get("results", [])
    ) if race.get("results") else len(laps)

    header_needed = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["season", "round", "circuit", "passes", "laps"])
        writer.writerow([season, round_no, circuit, passes, laps_completed])


if __name__ == "__main__":
    current_year = datetime.now().year
    fetch_data(2022, current_year)
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
