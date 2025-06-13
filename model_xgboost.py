"""
XGBoost binary classifier for predicting top‑3 finishes in Formula 1.

Usage:
    python model_xgboost.py  # prints fold metrics and overall mean ± std

Requirements:
    pip install xgboost scikit‑learn pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# 1. Load data
csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(csv_path)

# 2. Target & groups
df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

# 3. Feature matrix
cat_cols = ["circuit_id", "driver_id", "constructor_id"]
for col in cat_cols:
    df[col] = pd.Categorical(df[col]).codes

X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values

# 4. Cross‑validation
metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc"]}
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        tree_method="hist",  # change to 'gpu_hist' if GPU present
        random_state=42,
        verbosity=0,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_prob)

    metrics["acc"].append(acc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["f1"].append(f1)
    metrics["auc"].append(auc)

    print(
        f"[Fold {fold}] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  auc={auc:.3f}"
    )

print("\n=== XGBoost Results (mean ± std) ===")
for m, vals in metrics.items():
    print(f"{m}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")