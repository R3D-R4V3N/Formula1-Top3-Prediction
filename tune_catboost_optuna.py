"""
Optuna hyper‑parameter tuning for CatBoost predicting Formula 1 top‑3 finishes.

Run:
    python tune_catboost_optuna.py --trials 100 --threshold 0.45 --gpu

Arguments:
    --trials     Number of Optuna trials (default: 50)
    --threshold  Decision threshold for F1 calculation (default: 0.50)
    --gpu        Use GPU training (requires CUDA‑enabled CatBoost)

Requirements:
    pip install optuna catboost scikit‑learn pandas numpy
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# ---------------- CLI ---------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for F1")
parser.add_argument("--gpu", action="store_true", help="Enable GPU training")
args = parser.parse_args()

# ---------------- Data ---------------- #
csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(csv_path)

df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

gkf = GroupKFold(n_splits=5)

# ---------------- Objective ---------------- #

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1500, 6000),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_int("l2", 2, 10),
        "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "verbose": False,
        "class_weights": [1.0, (y == 0).sum() / (y == 1).sum()],
        "task_type": "GPU" if args.gpu else "CPU",
        "devices": "0" if args.gpu else None,
    }

    f1_scores = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_idx)

        model = CatBoostClassifier(**{k: v for k, v in params.items() if v is not None})
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=300)

        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= args.threshold).astype(int)
        f1_scores.append(f1_score(y[test_idx], pred))

    return float(np.mean(f1_scores))

# ---------------- Study ---------------- #
print(f"Starting Optuna tuning: {args.trials} trials, threshold={args.threshold}, GPU={args.gpu}")

study = optuna.create_study(direction="maximize", study_name="catboost_top3")
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

print("\nBest F1:", study.best_value)
print("Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
