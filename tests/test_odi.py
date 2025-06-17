import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import process_data
import compute_odi


def test_process_data_uses_previous_year(tmp_path, monkeypatch):
    odi_dir = tmp_path / "odi_cache"
    odi_dir.mkdir()
    with open(odi_dir / "odi_2021.json", "w") as f:
        json.dump({"silverstone": 0.3}, f)

    sample = {
        "circuit_id": "silverstone",
        "results": [
            {
                "Driver": {"driverId": "ham"},
                "Constructor": {"constructorId": "mer"},
                "position": "1",
                "status": "Finished",
            }
        ],
        "driver_standings": [{"Driver": {"driverId": "ham"}, "points": "25", "position": "1"}],
        "constructor_standings": [{"Constructor": {"constructorId": "mer"}, "points": "25", "position": "1"}],
        "qualifying": [{"Driver": {"driverId": "ham"}, "Q1": "1:20.0", "position": "1"}],
        "pitstops": [],
        "laps": [],
    }

    def fake_fetch(season, rnd):
        if season == 2022 and rnd == 1:
            return sample
        return None

    monkeypatch.setattr(process_data, "fetch_round_data", fake_fetch)
    monkeypatch.setattr(process_data, "log", lambda msg: None)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        process_data.prepare_dataset(2022, 2022, "out.csv")
    finally:
        os.chdir(cwd)

    with open(tmp_path / "out.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)

    assert header[-2:] == ["odi_raw", "grid_odi_mult"]
    assert row[-2:] == ["0.3", "0.3"]


def test_compute_odi_output(tmp_path, monkeypatch):
    laps = [
        {"number": "1", "Timings": [{"driverId": "a", "position": "1"}, {"driverId": "b", "position": "2"}]},
        {"number": "2", "Timings": [{"driverId": "b", "position": "1"}, {"driverId": "a", "position": "2"}]},
    ]
    sample = {
        "circuit_id": "foo",
        "results": [],
        "pitstops": [],
        "laps": laps,
    }

    def fake_fetch(season, rnd):
        if rnd == 1:
            return sample
        return None

    monkeypatch.setattr(compute_odi, "fetch_round_data", fake_fetch)
    monkeypatch.setattr(compute_odi, "log", lambda msg: None)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        compute_odi.compute_odi(2020, 2021)
    finally:
        os.chdir(cwd)

    out_file = tmp_path / "odi_cache" / "odi_2021.json"
    with open(out_file) as f:
        data = json.load(f)

    assert data["foo"] == 0.0

