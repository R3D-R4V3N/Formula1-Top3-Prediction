"""Streamlit web app for F1 podium prediction."""

from pathlib import Path

import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier, Pool

from model_catboost_final import MODEL_PARAMS
from predict_top3 import build_features


@st.cache_data
def load_dataset():
    """Load the processed racing dataset."""
    csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
    df = pd.read_csv(csv_path)
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
    df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train the CatBoost model on the full dataset."""
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
    return model, X.columns.tolist(), cat_idx


def main() -> None:
    st.title("F1 Podium Prediction")
    st.write(
        "Predict which drivers will finish on the podium for a selected race and explore feature importance."
    )

    df = load_dataset()
    model, feature_order, cat_idx = train_model(df)

    seasons = sorted(df["season_year"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    rounds = sorted(df[df["season_year"] == season]["round_number"].unique())
    round_no = st.selectbox("Round", rounds, index=len(rounds) - 1)

    if st.button("Predict top 3"):
        hist_df = df[
            (df["season_year"] < season)
            | ((df["season_year"] == season) & (df["round_number"] < round_no))
        ]
        features = build_features(season, round_no, hist_df)
        features = features[feature_order]
        preds = model.predict_proba(Pool(features, cat_features=cat_idx))[:, 1]
        features["probability"] = preds
        st.subheader("Predicted probabilities")
        st.dataframe(features.sort_values("probability", ascending=False))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Pool(features, cat_features=cat_idx))
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if shap_values.shape[1] == len(feature_order) + 1:
            shap_values = shap_values[:, :-1]

        st.subheader("Global feature importance")
        shap.summary_plot(shap_values, features[feature_order], show=False)
        st.pyplot(bbox_inches="tight")

        st.subheader("Driver explanations")
        for idx, row in (
            features.sort_values("probability", ascending=False).head(3).iterrows()
        ):
            st.markdown(f"### {row['driver_id']} – Prob: {row['probability']:.3f}")
            exp = shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=row[feature_order],
                feature_names=feature_order,
            )
            shap.plots.waterfall(exp, show=False)
            st.pyplot(bbox_inches="tight")


if __name__ == "__main__":
    main()
