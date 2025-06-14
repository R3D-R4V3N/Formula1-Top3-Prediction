from datetime import datetime
from pathlib import Path
import argparse

from fetch_data import fetch_data
from process_data import prepare_dataset


def refresh_lap_cache() -> None:
    """Remove cached lap data files."""
    cache_dir = Path("jolpica_f1_cache")
    for file in cache_dir.rglob("*_laps.json"):
        file.unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-laps",
        action="store_true",
        help="Invalidate cached lap data before fetching",
    )
    args = parser.parse_args()

    if args.refresh_laps:
        refresh_lap_cache()

    current_year = datetime.now().year
    fetch_data(2022, current_year)
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
