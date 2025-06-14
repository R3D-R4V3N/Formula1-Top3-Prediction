"""
Optuna hyper-parameter tuning voor CatBoost (max-CPU).
– Gebruikt alle cores   –  n_jobs=-1 voor parallelle trials
Runvoorbeeld:
    python tune_catboost_optuna_cpu_max.py --trials 300 --threshold 0.50
"""

import argparse, optuna, numpy as np, pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=300)
parser.add_argument('--threshold', type=float, default=0.50)
args = parser.parse_args()

# ---------- Data ----------
df = pd.read_csv(Path(__file__).with_name('f1_data_2022_to_present.csv'))
df['top3_flag'] = (df.finishing_position <= 3).astype(int)
df['group'] = df.season_year.astype(str) + '-' + df.round_number.astype(str)

X = df.drop(columns=['finishing_position', 'top3_flag', 'group'])
y = df.top3_flag.values
groups = df.group.values
cat_idx = [X.columns.get_loc(c) for c in ['circuit_id', 'driver_id', 'constructor_id']]
gkf = GroupKFold(5)

# ---------- Objective ----------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 1000, 5000),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.06, log=True),
        'depth': trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg': trial.suggest_int('l2', 2, 10),
        'bagging_temperature': trial.suggest_float('bag_temp', 0.0, 1.0),
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'thread_count': -1,             # gebruik alle CPU-cores
        'class_weights': [1.0, (y == 0).sum() / (y == 1).sum()],
    }
    thr = trial.suggest_float('thr',
                              args.threshold - 0.1,
                              args.threshold + 0.1)

    f1s = []
    for tr, te in gkf.split(X, y, groups):
        model = CatBoostClassifier(**params)
        model.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
                  eval_set=Pool(X.iloc[te], y[te], cat_features=cat_idx),
                  early_stopping_rounds=300)
        probs = model.predict_proba(X.iloc[te])[:, 1]
        f1s.append(f1_score(y[te], probs >= thr))
    return np.mean(f1s) - 0.5 * np.std(f1s)   # stabiel + hoog gemiddelde

# ---------- Optuna study ----------
study = optuna.create_study(direction='maximize')
study.optimize(objective,
               n_trials=args.trials,
               n_jobs=-1,          # parallel op alle cores
               show_progress_bar=True)

print('Best F1:', study.best_value)
print('Best parameters:')
print(study.best_params)
