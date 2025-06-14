"""
Scan decision thresholds (0.35–0.55) for the tuned CatBoost model and report per‑fold F1, precision, recall. 
Run after training to pick the global threshold that maximises mean F1.

Usage
-----
python threshold_scan_catboost.py

Requirements
------------
pip install catboost scikit‑learn pandas numpy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support

# ---------- tuned hyper‑parameters ----------
BEST_PARAMS = dict(
    iterations=1920,
    learning_rate=0.0339112437318613,
    depth=7,
    l2_leaf_reg=4,
    bagging_temperature=0.14256884085474172,
    random_seed=42,
    loss_function="Logloss",
    eval_metric="Logloss",   # keeps everything on GPU/CPU consistent
    verbose=False,
    class_weights=None,       # will be set after loading y
)

CSV = Path(__file__).with_name("f1_data_2022_to_present.csv")

# ---------- data ----------
df = pd.read_csv(CSV)
df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values
CAT_COLS = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

if BEST_PARAMS["class_weights"] is None:
    BEST_PARAMS["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

# ---------- threshold grid ----------
THR_GRID = np.arange(0.35, 0.56, 0.02)

# ---------- cross‑validation ----------
gkf = GroupKFold(n_splits=5)
results = {thr: [] for thr in THR_GRID}

for train_idx, test_idx in gkf.split(X, y, groups):
    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    test_pool  = Pool(X.iloc[test_idx],  y[test_idx],  cat_features=cat_idx)

    model = CatBoostClassifier(**BEST_PARAMS)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=200)

    probs = model.predict_proba(test_pool)[:, 1]
    for thr in THR_GRID:
        preds = (probs >= thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y[test_idx], preds, average="binary", zero_division=0)
        results[thr].append((pr, rc, f1))

# ---------- aggregate ----------
print("Threshold  Prec  Recall   F1")
best_thr, best_f1 = None, -1.0
for thr in THR_GRID:
    arr = np.array(results[thr])
    pr, rc, f1 = arr.mean(axis=0)
    print(f" {thr:0.2f}   {pr:0.3f}  {rc:0.3f}  {f1:0.3f}")
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print("\nBest threshold:", best_thr, "with mean F1:", round(best_f1, 3))
