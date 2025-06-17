"""Utilities for downloading and caching Jolpica F1 race data."""

import json
import os
import time
from datetime import datetime, timedelta

import requests
from meteostat import Hourly, Point
try:
    from pyowm.owm import OWM
except Exception:  # pragma: no cover - optional dependency may be missing
    OWM = None
from pytz import timezone

BASE_URL = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = "jolpica_f1_cache"
WEATHER_DIR = "weather_cache"


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
    """Fetch JSON from the Jolpica F1 API and return the 'MRData' section."""
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


def get_round_info(season: int, round_no: int):
    """Return the race info for a given season and round."""
    url = f"{BASE_URL}/{season}/{round_no}.json"
    data = fetch_json(url)
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0]
    return {}


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


def get_pitstops(season: int, round_no: int):
    """Return pit stop data for a given round."""
    # Use the `.json` endpoint with an explicit limit to avoid pagination
    url = f"{BASE_URL}/{season}/{round_no}/pitstops.json?limit=200"
    data = fetch_json(url)
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0].get("PitStops", [])
    return []


def get_lap_timings(season: int, round_no: int):
    """Return lap timing data for a given round."""
    url = f"{BASE_URL}/{season}/{round_no}/laps.json?limit=2000"
    data = fetch_json(url)
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0].get("Laps", [])
    return []


def fetch_weather(season: int, round_no: int):
    """Fetch and cache weather data for the given race."""
    os.makedirs(WEATHER_DIR, exist_ok=True)
    cache_file = os.path.join(WEATHER_DIR, f"weather_{season}_{round_no}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f)

    race_info = get_round_info(season, round_no)
    if not race_info:
        return {}

    loc = race_info.get("Circuit", {}).get("Location", {})
    lat = float(loc.get("lat"))
    lon = float(loc.get("long"))
    date_str = race_info.get("date")
    race_day = datetime.fromisoformat(date_str)

    local = timezone("Europe/Brussels")
    start = (
        local.localize(datetime(race_day.year, race_day.month, race_day.day, 12, 0))
        .astimezone(timezone("UTC"))
        .replace(tzinfo=None)
    )
    end = (
        local.localize(datetime(race_day.year, race_day.month, race_day.day, 14, 0))
        .astimezone(timezone("UTC"))
        .replace(tzinfo=None)
    )

    now = datetime.utcnow()
    features = {
        "temp_mean": None,
        "precip_sum": None,
        "humidity_mean": None,
        "wind_mean": None,
    }

    try:
        if start <= now:
            # historical average of the forecast window from the last ten years
            location = Point(lat, lon)
            temps, prcps, hums, winds = [], [], [], []
            for yr in range(start.year - 10, start.year):
                q_start = start.replace(year=yr) - timedelta(days=1)
                q_end = end.replace(year=yr) - timedelta(days=1)
                df_w = Hourly(location, q_start, q_end).fetch()
                if not df_w.empty:
                    temps.append(df_w["temp"].mean())
                    prcps.append(df_w["prcp"].sum())
                    hums.append(df_w["rhum"].mean())
                    winds.append(df_w["wspd"].mean())
            if temps:
                features = {
                    "temp_mean": float(sum(temps) / len(temps)),
                    "precip_sum": float(sum(prcps) / len(prcps)),
                    "humidity_mean": float(sum(hums) / len(hums)),
                    "wind_mean": float(sum(winds) / len(winds)),
                }
        else:
            api_key = os.getenv("OWM_API_KEY")
            if OWM and api_key:
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
                        "precip_sum": sum(prcps) / len(prcps) if prcps else None,
                        "humidity_mean": sum(hums) / len(hums) if hums else None,
                        "wind_mean": sum(winds) / len(winds) if winds else None,
                    }
    except Exception:
        pass

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(features, f)
    return features


def fetch_round_data(season: int, round_no: int):
    """Fetch and cache raw data for a given season and round."""
    os.makedirs(os.path.join(CACHE_DIR, str(season)), exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, str(season), f"{round_no}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            data = json.load(f)
        # Add newly introduced fields if missing
        changed = False
        if "pitstops" not in data:
            data["pitstops"] = get_pitstops(season, round_no)
            changed = True
        if "laps" not in data:
            data["laps"] = get_lap_timings(season, round_no)
            changed = True
        if changed:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        return data

    circuit_id, results = get_results(season, round_no)
    if not results:
        return None

    driver_standings = get_driver_standings(season, round_no)
    cons_standings = get_constructor_standings(season, round_no)
    qual_results = get_qualifying_results(season, round_no)
    pitstops = get_pitstops(season, round_no)
    laps = get_lap_timings(season, round_no)

    data = {
        "circuit_id": circuit_id,
        "results": results,
        "driver_standings": driver_standings,
        "constructor_standings": cons_standings,
        "qualifying": qual_results,
        "pitstops": pitstops,
        "laps": laps,
    }

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def fetch_data(start_season: int, end_season: int):
    """Download and cache raw data between seasons."""
    log(f"🏁 Fetching raw data from {start_season} to {end_season}")
    for season in range(start_season, end_season + 1):
        round_no = 1
        while True:
            log(f"🚦 {season} round {round_no}")
            data = fetch_round_data(season, round_no)
            if data is None:
                break
            fetch_weather(season, round_no)
            log(f"✅ cached data for {season} round {round_no}")
            round_no += 1


if __name__ == "__main__":
    current_year = datetime.now().year
    fetch_data(2022, current_year)
