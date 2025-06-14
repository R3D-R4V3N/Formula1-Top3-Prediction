"""
GPU‑friendly Optuna tuner – CatBoost top‑3 Formula 1.
----------------------------------------------------
Run‑voorbeeld:
    python tune_catboost_optuna_gpu.py --trials 500 --gpu

Argumenten
----------
--trials        Aantal Optuna‑trials (default 100)
--gpu           Gebruik CUDA‑GPU (task_type="GPU", devices="0")
--metric_freq   CatBoost logging‑interval (default 50)

Vereisten
---------
    pip install "catboost[gpu]" optuna scikit‑learn pandas numpy
"""

import argparse, optuna, numpy as np, pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# -------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=100)
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--metric_freq", type=int, default=50)
args = parser.parse_args()

# -------------- Data ---------------
df = pd.read_csv(Path(__file__).with_name("f1_data_2022_to_present.csv"))
df["top3_flag"] = (df.finishing_position <= 3).astype(int)
df["group"] = df.season_year.astype(str) + "-" + df.round_number.astype(str)
X = df.drop(columns=["finishing_position", "top3_flag", "group"])
y = df.top3_flag.values
groups = df.group.values
cat_cols = ["circuit_id", "driver_id", "constructor_id"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]
cv = GroupKFold(5)

# -------------- Objective ----------

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1500, 6000),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_int("l2", 2, 10),
        "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": 42,
        "verbose": False,
        "metric_period": args.metric_freq,
        "class_weights": [1.0, (y == 0).sum() / (y == 1).sum()],
        "allow_writing_files": False,
    }
    if args.gpu:
        params.update({"task_type": "GPU", "devices": "0"})

    thr = trial.suggest_float("thr", 0.40, 0.55)
    f1s = []
    for tr, te in cv.split(X, y, groups):
        model = CatBoostClassifier(**params)
        model.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
                  eval_set=Pool(X.iloc[te], y[te], cat_features=cat_idx),
                  early_stopping_rounds=400)
        p = model.predict_proba(X.iloc[te])[:, 1]
        f1s.append(f1_score(y[te], p >= thr))

    mean_f1 = float(np.mean(f1s))
    penalised = mean_f1 - 0.5 * float(np.std(f1s))  # straf variantie
    trial.set_user_attr("threshold", thr)
    return penalised

# -------------- Run Study ----------
study = optuna.create_study(direction="maximize", study_name="cat_gpu")
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

print("Best penalised objective:", study.best_value)
print("Best F1:", study.user_attrs[study.best_trial.number]["threshold"],
      study.best_value + 0.5 * np.std([t.value for t in study.trials]))
print("Best parameters:\n", study.best_params)
print("Best threshold (attr):", study.best_trial.user_attrs["threshold"])