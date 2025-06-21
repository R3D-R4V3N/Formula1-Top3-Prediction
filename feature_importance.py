"""Generate global and driver-level feature importance CSVs using SHAP.

This script trains the final CatBoost model on the full dataset and
computes feature importance with SHAP. Global mean absolute SHAP values
for each feature are written to `feature_importance/global_feature_importance.csv`.
Average SHAP values per driver are written to
`feature_importance/driver_feature_importance.csv`.

Usage:
    python feature_importance.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

from model_catboost_final import MODEL_PARAMS


def main() -> None:
    """Train the model and output feature importance CSVs."""
    csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
    df = pd.read_csv(csv_path)

    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
    df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

    drop_cols = ["finishing_position", "top3_flag", "group"]
    if "dnf_flag" in df.columns:
        drop_cols.append("dnf_flag")

    X = df.drop(columns=drop_cols)
    y = df["top3_flag"].values

    cat_cols = ["circuit_id", "driver_id", "constructor_id"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    params = MODEL_PARAMS.copy()
    params["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

    model = CatBoostClassifier(**params)
    model.fit(Pool(X, y, cat_features=cat_idx))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X, cat_features=cat_idx))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.shape[1] == len(X.columns) + 1:
        shap_values = shap_values[:, :-1]

    out_dir = Path("feature_importance")
    out_dir.mkdir(exist_ok=True)

    mean_abs = np.abs(shap_values).mean(axis=0)
    global_imp = pd.DataFrame({"feature": X.columns, "importance": mean_abs})
    global_imp.sort_values("importance", ascending=False, inplace=True)
    global_imp.to_csv(out_dir / "global_feature_importance.csv", index=False)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df["driver_id"] = X["driver_id"].values
    driver_imp = shap_df.groupby("driver_id").mean()
    driver_imp.to_csv(out_dir / "driver_feature_importance.csv")

    print("Global feature importance written to", out_dir / "global_feature_importance.csv")
    print("Driver-level feature importance written to", out_dir / "driver_feature_importance.csv")


if __name__ == "__main__":
    main()
