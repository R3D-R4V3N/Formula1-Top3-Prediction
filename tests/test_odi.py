import json
import os
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import process_data


def fake_fetch_round_data(season, round_no):
    if season != 2022 or round_no != 1:
        return None
    return {
        "circuit_id": "test_circuit",
        "results": [
            {
                "Driver": {"driverId": "drv1"},
                "Constructor": {"constructorId": "team1"},
                "position": "1",
                "grid": "1",
            },
            {
                "Driver": {"driverId": "drv2"},
                "Constructor": {"constructorId": "team2"},
                "position": "2",
                "grid": "2",
            },
        ],
        "driver_standings": [],
        "constructor_standings": [],
        "qualifying": [
            {
                "Driver": {"driverId": "drv1"},
                "position": "1",
                "Q1": "1:30.0",
            },
            {
                "Driver": {"driverId": "drv2"},
                "position": "2",
                "Q1": "1:31.0",
            },
        ],
        "pitstops": [],
    }


def test_odi_lookup_no_leakage(tmp_path, monkeypatch):
    os.chdir(tmp_path)
    os.mkdir("odi_cache")
    with open("odi_cache/odi_2021.json", "w", encoding="utf-8") as f:
        json.dump({"test_circuit": 0.2}, f)
    with open("odi_cache/odi_2022.json", "w", encoding="utf-8") as f:
        json.dump({"test_circuit": 0.8}, f)

    monkeypatch.setattr(process_data, "fetch_round_data", fake_fetch_round_data)

    out_csv = tmp_path / "out.csv"
    process_data.prepare_dataset(2022, 2022, str(out_csv))
    df = pd.read_csv(out_csv)

    assert not df["odi_raw"].isna().any()
    assert not df["grid_odi_mult"].isna().any()

    assert (df["grid_odi_mult"] == df["starting_grid_position"] * df["odi_raw"]).all()
    assert df.loc[0, "odi_raw"] == 0.2
