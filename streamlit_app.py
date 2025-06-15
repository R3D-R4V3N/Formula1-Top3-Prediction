import pandas as pd
import streamlit as st
from pathlib import Path
from catboost import CatBoostClassifier, Pool

from predict_top3 import build_features
from model_catboost_final import MODEL_PARAMS


def load_history():
    csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
    df = pd.read_csv(csv_path)
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
    df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)
    return df


def train_model(train_df):
    X = train_df.drop(
        columns=[
            "finishing_position",
            "top3_flag",
            "group",
            "grid_penalty_places",
            "grid_penalty_flag",
            "grid_bonus_flag",
        ]
    )
    y = train_df["top3_flag"].values
    cat_cols = ["circuit_id", "driver_id", "constructor_id"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    params = MODEL_PARAMS.copy()
    params["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

    model = CatBoostClassifier(**params)
    pool = Pool(X, y, cat_features=cat_idx)
    model.fit(pool, verbose=False)

    imp = model.get_feature_importance(pool)
    feat_imp = pd.Series(imp, index=X.columns).sort_values(ascending=False)

    return model, cat_idx, feat_imp, X.columns


def predict_top3_streamlit(season: int, round_no: int, hist_df: pd.DataFrame):
    train_df = hist_df[(hist_df["season_year"] < season) |
                       ((hist_df["season_year"] == season) & (hist_df["round_number"] < round_no))]

    model, cat_idx, feat_imp, cols = train_model(train_df)

    features = build_features(season, round_no, train_df)
    features = features[cols]
    probs = model.predict_proba(Pool(features, cat_features=cat_idx))[:, 1]
    features["prob"] = probs
    return features.sort_values("prob", ascending=False), feat_imp


# ---------------------- Streamlit UI ----------------------
st.title("F1 Top 3 Predictor")

hist_df = load_history()
seasons = sorted(hist_df["season_year"].unique())
season = st.selectbox("Season", seasons)
max_round = int(hist_df[hist_df["season_year"] == season]["round_number"].max())
round_no = st.number_input("Round", min_value=1, max_value=max_round, step=1, value=max_round)

if st.button("Predict Podium"):
    with st.spinner("Running prediction..."):
        results, feat_imp = predict_top3_streamlit(season, round_no, hist_df)
    st.subheader("Predicted Top 3")
    st.table(results.head(3)[["driver_id", "prob"]])
    st.subheader("Feature Importance")
    st.bar_chart(feat_imp.head(10))
