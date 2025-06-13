import argparse
import csv
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Iterator, Tuple

import pandas as pd
from requests import Response
from requests_cache import CachedSession

BASE = "https://api.jolpica.com/ergast/f1"

# Cached HTTP session
session = CachedSession(
    "f1_cache",
    expire_after=86400,
    allowable_codes=(200, 304),
)

# Deque for rate limiting (stores timestamps of requests)
REQUEST_LOG: deque[float] = deque(maxlen=500)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def rate_limited_get(url: str, retries: int = 5) -> Response:
    """GET request respecting API limits and retrying on errors."""
    for attempt in range(retries):
        _respect_rate_limits()
        logging.info("GET %s", url)
        try:
            resp = session.get(url, timeout=10)
        except Exception as exc:  # requests.exceptions.RequestException
            delay = 2 ** attempt
            logging.warning("connection error: %s, retrying in %.1fs", exc, delay)
            time.sleep(delay)
            continue

        REQUEST_LOG.append(time.monotonic())
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 0))
            delay = max(retry_after, 2 ** attempt)
            logging.warning("429 received, retrying in %.1fs", delay)
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp

    resp.raise_for_status()
    return resp


def _respect_rate_limits() -> None:
    """Sleep if sending a new request would exceed the limits."""
    now = time.monotonic()
    if REQUEST_LOG:
        # 4 requests per second
        elapsed = now - REQUEST_LOG[-1]
        if elapsed < 0.25:
            time.sleep(0.25 - elapsed)
    if len(REQUEST_LOG) == REQUEST_LOG.maxlen:
        # 500 requests per hour
        elapsed = now - REQUEST_LOG[0]
        if elapsed < 3600:
            time.sleep(3600 - elapsed)


def fetch_results_page(season: int, page: int) -> Dict[str, Any]:
    url = f"{BASE}/{season}/results.json?limit=100&offset={page * 100}"
    resp = rate_limited_get(url)
    return resp.json().get("MRData", {})


def load_checkpoint() -> Tuple[int, int] | None:
    if not os.path.exists("checkpoint.json"):
        return None
    with open("checkpoint.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("season"), data.get("page")


def save_checkpoint(season: int, page: int) -> None:
    with open("checkpoint.json", "w", encoding="utf-8") as f:
        json.dump({"season": season, "page": page, "timestamp": time.time()}, f)


def iter_results(season: int, start_page: int = 0) -> Iterator[Tuple[int, list[Dict[str, Any]]]]:
    page = start_page
    while True:
        data = fetch_results_page(season, page)
        races = data.get("RaceTable", {}).get("Races", [])
        total = int(data.get("total", 0))
        if not races:
            break
        yield page, races
        page += 1
        if page * 100 >= total:
            break


def write_page(season: int, page: int, races: list[Dict[str, Any]], output_file: str) -> None:
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["season", "round", "raceId", "driverId", "grid", "position"])
        for race in races:
            round_no = race.get("round")
            race_id = f"{season}_{round_no}"
            for result in race.get("Results", []):
                writer.writerow([
                    season,
                    round_no,
                    race_id,
                    result.get("Driver", {}).get("driverId"),
                    result.get("grid"),
                    result.get("position"),
                ])
    save_checkpoint(season, page)


def collect_data(start_season: int, end_season: int, output_file: str) -> None:
    cp = load_checkpoint()
    for season in range(start_season, end_season + 1):
        start_page = 0
        if cp and cp[0] == season:
            start_page = cp[1] + 1
        for page, races in iter_results(season, start_page):
            write_page(season, page, races, output_file)
        cp = None  # reset after first use


def compute_net_gain_5(input_file: str) -> None:
    df = pd.read_csv(input_file)
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["NET"] = df["grid"] - df["position"]
    df.sort_values(["driverId", "raceId"], inplace=True)
    df["NET_GAIN_5"] = (
        df.groupby("driverId")["NET"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df[["driverId", "raceId", "NET_GAIN_5"]].to_csv("net_gain_5.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download F1 results")
    parser.add_argument("--start-season", type=int, required=True)
    parser.add_argument("--end-season", type=int, required=True)
    args = parser.parse_args()

    output = "f1_results.csv"
    collect_data(args.start_season, args.end_season, output)
    compute_net_gain_5(output)
