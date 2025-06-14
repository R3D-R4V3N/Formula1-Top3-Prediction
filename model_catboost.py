"""
CatBoost top‑3 podium predictor – tuned parameters (Optuna trial #93) and 0.50 decision threshold.

Usage
-----
python model_catboost_final.py   # prints 5‑fold GroupKFold metrics

Hyper‑parameters
----------------
iterations          = 1920
learning_rate       = 0.0339112437318613
depth               = 7
l2_leaf_reg         = 4
bagging_temperature = 0.14256884085474172
threshold           = 0.50

Requirements
------------
 pip install catboost scikit‑learn pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# ---------- 1. Load data ----------
CSV = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(CSV)

# ---------- 2. Target & group key ----------
df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# ---------- 3. Model params ----------
params = dict(
    iterations=1920,
    learning_rate=0.0339112437318613,
    depth=7,
    l2_leaf_reg=4,
    bagging_temperature=0.14256884085474172,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    class_weights=[1.0, (y == 0).sum() / (y == 1).sum()],
)

THRESHOLD = 0.50

# ---------- 4. Cross‑validation ----------
gkf = GroupKFold(n_splits=5)
metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc"]}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_indices)
    valid_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_indices)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=300)

    p = model.predict_proba(X.iloc[test_idx])[:, 1]
    y_pred = (p >= THRESHOLD).astype(int)

    acc = accuracy_score(y[test_idx], y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y[test_idx], y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y[test_idx], p)

    for k, v in zip(["acc", "prec", "rec", "f1", "auc"], [acc, prec, rec, f1, auc]):
        metrics[k].append(v)

    print(
        f"[Fold {fold}] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  "
        f"f1={f1:.3f}  auc={auc:.3f}"
    )

print("\n=== Tuned CatBoost Results (mean ± std) ===")
for m, vals in metrics.items():
    print(f"{m}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
