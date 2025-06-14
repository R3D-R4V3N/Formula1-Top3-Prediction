import pandas as pd


def test_monaco_harder_than_spa():
    df = pd.read_csv('f1_data_2022_to_present.csv')
    df = df.dropna(subset=['starting_grid_position', 'finishing_position'])
    df['delta'] = (df['finishing_position'] - df['starting_grid_position']).abs()
    pass_rates = df.groupby('circuit_id')['delta'].mean()
    min_p = pass_rates.min()
    max_p = pass_rates.max()
    diff = 1 - (pass_rates - min_p) / (max_p - min_p)
    assert diff['monaco'] > diff['spa']
