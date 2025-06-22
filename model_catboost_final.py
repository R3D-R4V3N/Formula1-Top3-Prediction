import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from group_time_series_split import GroupTimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression

# -------------------- Config --------------------
# Decision threshold tuned via threshold_scan_final.py --calibrate
THRESHOLD = 0.52

MODEL_PARAMS = dict(
    iterations=4864,
    learning_rate=0.04129221307953875,
    depth=8,
    l2_leaf_reg=10,
    bagging_temperature=0.7413031400854047,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

PARAMS_FILE = Path(__file__).with_name("optuna_best_params.json")
THRESHOLD_FILE = Path(__file__).with_name("best_threshold.json")

def _normalize_params(params: dict) -> dict:
    """Convert shorthand keys from older tuning outputs."""
    mapping = {
        "lr": "learning_rate",
        "l2": "l2_leaf_reg",
        "bag_temp": "bagging_temperature",
    }
    return {mapping.get(k, k): v for k, v in params.items()}

if PARAMS_FILE.exists():
    with open(PARAMS_FILE, encoding="utf-8") as f:
        loaded = json.load(f)
    params = _normalize_params(loaded.get("model_params", loaded))
    MODEL_PARAMS.update(params)
    if "threshold" in loaded:
        THRESHOLD = loaded["threshold"]

if THRESHOLD_FILE.exists():
    with open(THRESHOLD_FILE, encoding="utf-8") as f:
        THRESHOLD = json.load(f).get("threshold", THRESHOLD)

# -------------------- Load data --------------------
csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(csv_path)

if "top3_flag" not in df.columns:
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

drop_cols = ["finishing_position", "top3_flag", "group"]
if "dnf_flag" in df.columns:
    drop_cols.append("dnf_flag")

X = df.drop(columns=drop_cols)
y = df["top3_flag"].values

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

MODEL_PARAMS["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

# -------------------- Cross-validation --------------------
tscv = GroupTimeSeriesSplit(n_splits=5)
metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc"]}

# Pre-allocate for calibration
y_probs_all = np.zeros_like(y, dtype=float)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X, groups=df["group"]), 1):
    model = CatBoostClassifier(**MODEL_PARAMS)

    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[test_idx],  y[test_idx],  cat_features=cat_idx)

    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=300)
    y_probs_all[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

# -------------------- Platt calibration --------------------
calibrator = LogisticRegression(max_iter=1000)
calibrator.fit(y_probs_all.reshape(-1, 1), y)
y_probs_calibrated = calibrator.predict_proba(y_probs_all.reshape(-1, 1))[:, 1]

# -------------------- Evaluation --------------------
preds = (y_probs_calibrated >= THRESHOLD).astype(int)
acc = accuracy_score(y, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
auc = roc_auc_score(y, y_probs_calibrated)

print("\n=== Final Calibrated CatBoost Results (Recall-Focused) ===")
print(f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  auc={auc:.3f}")

