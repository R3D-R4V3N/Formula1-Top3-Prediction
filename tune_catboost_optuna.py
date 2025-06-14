"""
GPU‑friendly Optuna tuner for CatBoost – Formula 1 top‑3 prediction.

Run (RTX 4050 example, 500 trials, threshold scan centred on 0.50):
    python tune_catboost_optuna_gpu.py --trials 500 --gpu

Arguments
---------
--trials      Number of Optuna trials (default 100)
--threshold   Base decision threshold centre (default 0.50)
--gpu         Use CUDA (task_type="GPU")
--metric_freq CatBoost metric print frequency (default 50)

The objective maximises **mean F1 – 0.5·σ(F1)** across GroupKFold(5), so
Optuna favours both high accuracy and low fold variance. Threshold is
co‑optimised in [base‑0.05, base+0.05].

Dependencies
------------
    pip install catboost optuna pandas numpy scikit‑learn
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
parser = argparse.ArgumentParser(description="Optuna tuner for CatBoost F1‑score (GPU ready)")
parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
parser.add_argument("--threshold", type=float, default=0.50, help="Central decision threshold")
parser.add_argument("--gpu", action="store_true", help="Enable GPU training")
parser.add_argument("--metric_freq", type=int, default=50, help="CatBoost metric logging freq")
args = parser.parse_args()

# ---------------- Data ---------------- #
df = pd.read_csv(Path(__file__).with_name("f1_data_2022_to_present.csv"))

df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df["top3_flag"].values
groups = df["group"].values

cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

gkf = GroupKFold(n_splits=5)

# Pre‑compute class weight ratio (neg / pos)
scale_pos = (y == 0).sum() / (y == 1).sum()

# ---------------- Optuna Objective ---------------- #

def objective(trial):
    # Hyper‑parameter search space
    params = {
        "iterations": trial.suggest_int("iterations", 2000, 6000),
        "learning_rate": trial.suggest_float("lr", 0.015, 0.04, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_int("l2", 2, 12),
        "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
        "random_seed": 42,
        "loss_function": "Logloss",   # fully GPU compatible
        "eval_metric": "Logloss",
        "verbose": False,
        "metric_period": args.metric_freq,
        "allow_writing_files": False,
        "class_weights": [1.0, scale_pos],
        "task_type": "GPU" if args.gpu else "CPU",
        "devices": "0" if args.gpu else None,
    }

    # Threshold search ±0.05 rond basiswaarde
    thr = trial.suggest_float("thr", args.threshold - 0.05, args.threshold + 0.05)

    f1_list = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_idx)

        model = CatBoostClassifier(**{k: v for k, v in params.items() if v is not None})
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=300)

        prob = model.predict_proba(X.iloc[test_idx])[:, 1]
        pred = (prob >= thr).astype(int)
        f1_list.append(f1_score(y[test_idx], pred))

    # Optimise for mean F1 minus half the variance
    return float(np.mean(f1_list) - 0.5 * np.std(f1_list))

# ---------------- Run Study ---------------- #
print(f"\n[Optuna] trials={args.trials} | base thr={args.threshold:.2f} | GPU={args.gpu}\n")

study = optuna.create_study(direction="maximize", study_name="catboost_top3_gpu")
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

print("\nBest study value (mean F1 – 0.5·σ):", study.best_value)
print("Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
