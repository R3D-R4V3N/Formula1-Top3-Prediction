"""Compute Overtake Difficulty Index (ODI) from Jolpica F1 data."""

import argparse
import json
import os
from collections import defaultdict
from statistics import median

from fetch_data import fetch_round_data, get_lap_times, log


def overtakes_per_lap(season: int, round_no: int):
    """Return circuit id and overtakes per lap for a race."""
    data = fetch_round_data(season, round_no)
    if data is None:
        return None, None

    circuit = data["circuit_id"]
    pitstops = data.get("pitstops", [])
    laps = get_lap_times(season, round_no)
    if not laps:
        return circuit, None

    pit_windows = defaultdict(set)
    for p in pitstops:
        drv = p.get("driverId")
        try:
            lap = int(p.get("lap"))
        except (TypeError, ValueError):
            continue
        pit_windows[drv].add(lap)
        pit_windows[drv].add(lap + 1)

    prev_pos = {}
    overtakes = 0
    for lap in laps:
        try:
            lap_no = int(lap.get("number"))
        except (TypeError, ValueError):
            continue
        for t in lap.get("Timings", []):
            drv = t.get("driverId")
            try:
                pos = int(t.get("position"))
            except (TypeError, ValueError):
                continue
            if drv in prev_pos:
                if (
                    lap_no not in pit_windows.get(drv, set())
                    and lap_no - 1 not in pit_windows.get(drv, set())
                ):
                    diff = pos - prev_pos[drv]
                    if diff > 0:
                        overtakes += diff
            prev_pos[drv] = pos
    laps_count = len(laps)
    return circuit, (overtakes / laps_count if laps_count else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ODI cache")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    log(f"🔄 scanning overtakes {args.start}-{args.end}")
    raw = defaultdict(dict)
    for season in range(args.start, args.end + 1):
        round_no = 1
        while True:
            circ, opl = overtakes_per_lap(season, round_no)
            if circ is None:
                break
            if opl is not None:
                raw[circ][season] = opl
            round_no += 1

    all_opls = [v for circ in raw.values() for v in circ.values() if v is not None]
    if not all_opls:
        log("❌ no data found")
        return
    max_opl = max(all_opls)

    odi = {}
    for circ, seasons in raw.items():
        hist_seasons = [
            s for s in range(max(args.start, args.end - 4), args.end + 1) if s in seasons
        ]
        if not hist_seasons:
            continue
        med = median([seasons[s] for s in hist_seasons])
        val = 1.0 - (med / max_opl if max_opl else 0.0)
        val = max(0.0, min(1.0, val))
        odi[circ] = val

    os.makedirs("odi_cache", exist_ok=True)
    out_file = os.path.join("odi_cache", f"odi_{args.end}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(odi, f, indent=2)

    log(f"✅ wrote ODI for {len(odi)} circuits to {out_file}")


if __name__ == "__main__":
    main()
