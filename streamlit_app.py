import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from catboost import CatBoostClassifier, Pool
from model_catboost_final import MODEL_PARAMS

# -------------------- Config --------------------
THRESHOLD = 0.42  # recall-geoptimaliseerde threshold
DATA_PATH = Path(__file__).with_name("f1_data_2022_to_present.csv")
NATIONALITY_PATH = Path(__file__).with_name("nationality.json")

# -------------------- Load data --------------------
df = pd.read_csv(DATA_PATH)
with open(NATIONALITY_PATH, "r", encoding="utf-8") as f:
    NATIONALITY = json.load(f)

if "top3_flag" not in df.columns:
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
if "group" not in df.columns:
    df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

# Select race group (laatste of specifieke)
group_options = sorted(df["group"].unique())
selected_group = st.selectbox("Selecteer race (season-round)", options=group_options, index=len(group_options)-1)

race_df = df[df["group"] == selected_group].copy()
race_df["Flag"] = race_df["driver_id"].map(NATIONALITY).fillna("")
race_df["Driver"] = race_df["driver_id"] + " " + race_df["Flag"]
input_df = race_df.drop(columns=["finishing_position", "top3_flag", "group"], errors="ignore")

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [input_df.columns.get_loc(c) for c in cat_cols if c in input_df.columns]

# -------------------- Train model on full data --------------------
X_full = df.drop(columns=["finishing_position", "top3_flag", "group"], errors="ignore")
y_full = df["top3_flag"].values
model = CatBoostClassifier(**MODEL_PARAMS)
model.fit(X_full, y_full, cat_features=cat_idx, verbose=False)

# -------------------- Predict --------------------
input_pool = Pool(input_df, cat_features=cat_idx)
raw_probs = model.predict_proba(input_pool)[:, 1]
calibrated_probs = raw_probs
preds = (calibrated_probs >= THRESHOLD).astype(int)

# Resultaat verwerken
race_df["Podium kans"] = np.round(calibrated_probs * 100, 1)
race_df["Voorspelling"] = np.where(preds == 1, "Top 3", "Niet Top 3")
race_df["Waarde Tip"] = np.where(calibrated_probs >= THRESHOLD, "✅", "")

# Weergave
st.title("🏁 F1 Podium Predictie (Top 3)")
st.subheader(f"Resultaten voor race: {selected_group}")

st.dataframe(
    race_df[["Driver", "constructor_id", "Podium kans", "Voorspelling", "Waarde Tip"]]
    .sort_values("Podium kans", ascending=False)
    .reset_index(drop=True)
)

# Optioneel export
csv = race_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download resultaten (CSV)", csv, "race_voorspellingen.csv", "text/csv")
