"""
Threshold Scan for Tuned CatBoost Model
=======================================
Rescans decision thresholds for the current best CatBoost model to maximize F1
while monitoring recall. Accepts CLI args for threshold range.

Usage:
    python threshold_scan_final.py --start 0.50 --end 0.65 --step 0.02

Requirements:
    pandas, numpy, catboost, scikit-learn
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# ---------- CLI args ----------
parser = argparse.ArgumentParser(description="Threshold scan for CatBoost model")
parser.add_argument("--start", type=float, default=0.50,
                    help="starting threshold (inclusive)")
parser.add_argument("--end", type=float, default=0.65,
                    help="ending threshold (inclusive)")
parser.add_argument("--step", type=float, default=0.02,
                    help="threshold step size")
args = parser.parse_args()

# ---------- tuned hyper-parameters ----------
from model_catboost_final import MODEL_PARAMS  # includes class_weights and THRESHOLD

# ---------- load data ----------
csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
df = pd.read_csv(csv_path)
df = df.sort_values(['season_year', 'round_number']).reset_index(drop=True)
# ensure target and grouping
if 'top3_flag' not in df.columns:
    df['top3_flag'] = (df['finishing_position'] <= 3).astype(int)
if 'group' not in df.columns:
    df['group'] = df['season_year'].astype(str) + '-' + df['round_number'].astype(str)

X = df.drop(
    columns=[
        'finishing_position',
        'top3_flag',
        'group',
        'grid_penalty_places',
        'grid_penalty_flag',
        'grid_bonus_flag',
    ]
)
y = df['top3_flag'].values
groups = df['group'].values
unique_groups = df['group'].unique()
tscv = TimeSeriesSplit(n_splits=5)
cat_idx = [X.columns.get_loc(c) for c in ['circuit_id','driver_id','constructor_id']]

# ---------- collect OOF probabilities ----------
y_probs = np.zeros_like(y, dtype=float)

for tr_g_idx, te_g_idx in tscv.split(unique_groups):
    train_groups = unique_groups[tr_g_idx]
    test_groups = unique_groups[te_g_idx]
    train_idx = df.index[df['group'].isin(train_groups)]
    test_idx = df.index[df['group'].isin(test_groups)]
    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[test_idx],  y[test_idx],  cat_features=cat_idx)
    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=200)
    y_probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:,1]

# ---------- threshold scan ----------
thresholds = np.arange(args.start, args.end + args.step/2, args.step)
results = []
for thr in thresholds:
    preds = (y_probs >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
    auc = roc_auc_score(y, y_probs)
    results.append((thr, precision, recall, f1, auc))

df_res = pd.DataFrame(results, columns=['threshold','precision','recall','f1','auc'])
print(df_res.to_string(index=False))

best = df_res[df_res['f1']==df_res['f1'].max()].iloc[0]
print(f"\nBest threshold: {best.threshold:.2f} -> F1 = {best.f1:.3f}, Recall = {best.recall:.3f}")
