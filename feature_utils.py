import pandas as pd
from pathlib import Path


def add_overtaking_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overtaking difficulty by circuit_id and fill missing values."""
    csv_path = Path(__file__).with_name("overtaking_difficulty.csv")
    odf = pd.read_csv(csv_path)
    df = df.merge(
        odf[["circuit_id", "overtaking_difficulty"]],
        on="circuit_id",
        how="left",
    )
    median_value = odf["overtaking_difficulty"].median()
    df["overtaking_difficulty"] = df["overtaking_difficulty"].fillna(median_value)
    return df
