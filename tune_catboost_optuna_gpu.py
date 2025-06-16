"""
GPU‑friendly Optuna tuner voor CatBoost – F1‑σ omgeving.

Run:
    python tune_catboost_optuna_gpu.py --trials 500 --gpu
"""

import argparse, optuna, numpy as np, pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=300)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

# ---------- Data ----------
df = pd.read_csv(Path(__file__).with_name('f1_data_2022_to_present.csv'))
df['top3_flag'] = (df.finishing_position <= 3).astype(int)
df['group'] = df.season_year.astype(str) + '-' + df.round_number.astype(str)
drop_cols = ['finishing_position', 'top3_flag', 'group']
if 'dnf_flag' in df.columns:
    drop_cols.append('dnf_flag')
X = df.drop(columns=drop_cols)
y = df.top3_flag.values
cat_idx = [X.columns.get_loc(c) for c in ['circuit_id', 'driver_id', 'constructor_id']]
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1500, 6000),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_int('l2', 2, 10),
        'bagging_temperature': trial.suggest_float('bag_temp', 0.0, 1.0),
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',   # pure GPU metric
        'random_seed': 42,
        'verbose': False,
        'metric_period': 50,
        'allow_writing_files': False,
        'class_weights': [1.0, (y == 0).sum()/ (y == 1).sum()],
    }
    if args.gpu:
        params.update({'task_type': 'GPU', 'devices': '0'})

    thr = trial.suggest_float('thr', 0.40, 0.55)
    f1s = []
    for tr, te in tscv.split(X):
        model = CatBoostClassifier(**params)
        model.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
                  eval_set=Pool(X.iloc[te], y[te], cat_features=cat_idx),
                  early_stopping_rounds=300)
        probs = model.predict_proba(X.iloc[te])[:, 1]
        f1s.append(f1_score(y[te], probs >= thr))
    return np.mean(f1s) - 0.5*np.std(f1s)   # stabiel + hoog gemiddelde

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
print('Best study value:', study.best_value)
print('Best parameters:')
print(study.best_params)