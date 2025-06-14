import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fetch_data import get_laps, get_pitstops, get_status
from process_data import detect_passes, circuit_difficulty, race_pass_df
from model_catboost_final import main as train_model


@pytest.mark.parametrize(
    "season,round_no,expected,delta",
    [
        (2023, 7, 36, 5),
        (2023, 1, 80, 10),
    ],
)
def test_pass_counts(season, round_no, expected, delta):
    laps = get_laps(season, round_no)
    pits = get_pitstops(season, round_no)
    status = get_status(season, round_no)
    passes = detect_passes(laps, pits, status)
    assert abs(passes - expected) <= delta


def test_shap_importance():
    train_model()


