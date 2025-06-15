"""Generate SHAP feature importance plots for the CatBoost model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool

from model_catboost_final import MODEL_PARAMS


def load_data() -> tuple[pd.DataFrame, pd.Series, list[int]]:
    csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
    df = pd.read_csv(csv_path)
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)

    drop_cols = ["finishing_position", "top3_flag"]
    if "dnf_flag" in df.columns:
        drop_cols.append("dnf_flag")

    X = df.drop(columns=drop_cols)
    y = df["top3_flag"]
    cat_cols = ["circuit_id", "driver_id", "constructor_id"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    return X, y, cat_idx


def train_model(X: pd.DataFrame, y: pd.Series, cat_idx: list[int]) -> CatBoostClassifier:
    params = MODEL_PARAMS.copy()
    params["class_weights"] = [1.0, (y == 0).sum() / max((y == 1).sum(), 1)]
    model = CatBoostClassifier(**params)
    model.fit(Pool(X, y, cat_features=cat_idx))
    return model


def shap_importance(model: CatBoostClassifier, X: pd.DataFrame, cat_idx: list[int]):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Pool(X, cat_features=cat_idx))

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.show()

    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.show()


def main() -> None:
    X, y, cat_idx = load_data()
    model = train_model(X, y, cat_idx)
    shap_importance(model, X, cat_idx)


if __name__ == "__main__":
    main()

