import pandas as pd


def test_csv_keys_are_strings():
    df = pd.read_csv('circuit_overtake_difficulty.csv')
    assert all(not str(cid).startswith('(') for cid in df['circuit_id'])
