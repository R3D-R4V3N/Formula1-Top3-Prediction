"""
CatBoost binary classifier for predicting top‑3 finishes in Formula 1.

Usage:
    python model_catboost.py  # prints fold metrics and overall mean ± std

Requirements:
    pip install catboost scikit‑learn pandas numpy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# 1. Load data
csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(csv_path)

# 2. Target & groups
df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

# 3. Features & categorical indices
X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_indices = [X.columns.get_loc(c) for c in cat_cols]

# 4. GroupKFold cross‑validation
gkf = GroupKFold(n_splits=5)
metrics = {k: [] for k in ["acc", "prec", "rec", "f1", "auc"]}

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    valid_pool = Pool(X_test, y_test, cat_features=cat_indices)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=False,
        class_weights=[1.0, (y == 0).sum() / (y == 1).sum()],
    )

    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)

    metrics["acc"].append(acc)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["f1"].append(f1)
    metrics["auc"].append(auc)

    print(f"[Fold {fold}] acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  auc={auc:.3f}")

print("\n=== CatBoost Results (mean ± std) ===")
for m, vals in metrics.items():
    print(f"{m}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")