import argparse
import json
import os
import statistics

from fetch_data import fetch_round_data, log


def compute_odi(start: int, end: int) -> None:
    log(f"🔄 scanning overtakes {start}-{end}")
    raw = {}
    max_opl = 0.0
    for season in range(start, end + 1):
        round_no = 1
        while True:
            data = fetch_round_data(season, round_no)
            if data is None:
                break
            laps = data.get("laps", [])
            if not laps:
                round_no += 1
                continue
            pitstops = data.get("pitstops", [])
            pit_laps = {}
            for p in pitstops:
                drv = p.get("driverId")
                lap = p.get("lap")
                if drv and lap:
                    pit_laps.setdefault(drv, set()).add(int(lap))

            positions = {}
            for lap in laps:
                num = int(lap.get("number"))
                positions[num] = {t["driverId"]: int(t["position"]) for t in lap.get("Timings", [])}

            lap_nums = sorted(positions)
            overtakes = 0
            for i in range(1, len(lap_nums)):
                prev = positions[lap_nums[i - 1]]
                curr = positions[lap_nums[i]]
                lap_id = lap_nums[i]
                for drv, pos in curr.items():
                    if lap_id in pit_laps.get(drv, set()):
                        continue
                    prev_pos = prev.get(drv)
                    if prev_pos is None:
                        continue
                    delta = prev_pos - pos
                    if delta > 0:
                        overtakes += delta
            laps_total = lap_nums[-1] if lap_nums else 0
            if laps_total > 0:
                opl = overtakes / laps_total
                circ = data["circuit_id"]
                raw.setdefault(circ, {})[season] = opl
                max_opl = max(max_opl, opl)
            round_no += 1

    hist_start = max(2020, end - 4)
    seasons_hist = list(range(hist_start, end + 1))
    lookup = {}
    for circ, by_season in raw.items():
        vals = [by_season[s] for s in seasons_hist if s in by_season]
        if not vals:
            continue
        med = statistics.median(vals)
        odi_raw = 1 - (med / max_opl) if max_opl else 0.0
        lookup[circ] = odi_raw

    os.makedirs("odi_cache", exist_ok=True)
    out_file = os.path.join("odi_cache", f"odi_{end}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)
    log(f"✅ wrote ODI for {len(lookup)} circuits to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    compute_odi(args.start, args.end)
