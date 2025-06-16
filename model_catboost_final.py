"""
Final CatBoost model for predicting top‑3 podium finishes in Formula 1.

Performance target: F1 ≥ 0.80 and recall ≥ 0.90 (5‑fold TimeSeriesSplit)

Usage:
    python model_catboost_final.py

Dependencies:
    pip install catboost scikit‑learn pandas numpy


Best trial: 202. Best value: 0.90603: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 400/400 [2:34:17<00:00, 23.14s/it]
Best F1: 0.9060295851914649
Best parameters:
{'iterations': 1528, 'lr': 0.016285023353888272, 'depth': 6, 'l2': 9, 'bag_temp': 0.5922072238932671, 'thr': 0.6499789940017192}
jasper@jasper-XPS-15-9530:~/Documents/Github/Formula1-Top3-Prediction$ 
"""

import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# -------------------- Config --------------------
THRESHOLD = 0.6499789940017192  # tuned decision cutoff

MODEL_PARAMS = dict(
    iterations=1528,
    learning_rate=0.016285023353888272,
    depth=6,
    l2_leaf_reg=9,
    bagging_temperature=0.5922072238932671,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
)

# -------------------- Load data --------------------
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

MODEL_PARAMS["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

# -------------------- Cross‑validation --------------------
tscv = TimeSeriesSplit(n_splits=5)
metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc"]}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    model = CatBoostClassifier(**MODEL_PARAMS)

    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[test_idx],  y[test_idx],  cat_features=cat_idx)

    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=300)

    probs = model.predict_proba(X.iloc[test_idx])[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    acc = accuracy_score(y[test_idx], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y[test_idx], preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(y[test_idx], probs)

    metrics["acc"].append(acc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["f1"].append(f1)
    metrics["auc"].append(auc)

    print(
        f"[Fold {fold}] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  "
        f"f1={f1:.3f}  auc={auc:.3f}"
    )

print("\n=== Tuned CatBoost Results (mean ± std) ===")
for m, vals in metrics.items():
    print(f"{m}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
