import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from group_time_series_split import GroupTimeSeriesSplit
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from model_catboost_final import MODEL_PARAMS


def _normalize_params(params: dict) -> dict:
    """Convert shorthand keys from older tuning outputs."""
    mapping = {
        "lr": "learning_rate",
        "l2": "l2_leaf_reg",
        "bag_temp": "bagging_temperature",
    }
    return {mapping.get(k, k): v for k, v in params.items()}

# ---------- CLI args ----------
parser = argparse.ArgumentParser(description="Threshold scan for CatBoost model")
parser.add_argument("--start", type=float, default=0.50, help="starting threshold (inclusive)")
parser.add_argument("--end", type=float, default=0.65, help="ending threshold (inclusive)")
parser.add_argument("--step", type=float, default=0.02, help="threshold step size")
parser.add_argument("--calibrate", action="store_true", help="apply Platt calibration")
parser.add_argument("--save", type=str, default=None, help="optional path to save results as CSV")
parser.add_argument("--data", type=str, default='f1_data_2022_to_present.csv')
parser.add_argument("--params", type=str, default=None, help="JSON with model parameters")
parser.add_argument("--save-json", type=str, default=None, help="optional path to save best threshold as JSON")
args = parser.parse_args()

if args.params:
    with open(args.params, encoding='utf-8') as f:
        loaded = json.load(f)
    params = _normalize_params(loaded.get('model_params', loaded))
    MODEL_PARAMS.update(params)

# ---------- load data ----------
df = pd.read_csv(Path(args.data))

if 'top3_flag' not in df.columns:
    df['top3_flag'] = (df['finishing_position'] <= 3).astype(int)
if 'group' not in df.columns:
    df['group'] = df['season_year'].astype(str) + '-' + df['round_number'].astype(str)

X = df.drop(columns=['finishing_position', 'top3_flag', 'group'])
y = df['top3_flag'].values
cat_idx = [X.columns.get_loc(c) for c in ['circuit_id', 'driver_id', 'constructor_id']]

# ---------- collect OOF probabilities ----------
y_probs = np.zeros_like(y, dtype=float)

tscv = GroupTimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X, groups=df['group']):
    train_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[test_idx],  y[test_idx],  cat_features=cat_idx)
    model = CatBoostClassifier(**MODEL_PARAMS)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=200)
    y_probs[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

# ---------- optional calibration ----------
if args.calibrate:
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(y_probs.reshape(-1, 1), y)
    y_probs = calibrator.predict_proba(y_probs.reshape(-1, 1))[:, 1]

# ---------- threshold scan ----------
thresholds = np.arange(args.start, args.end + args.step / 2, args.step)
results = []
for thr in thresholds:
    preds = (y_probs >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
    auc = roc_auc_score(y, y_probs)
    results.append((thr, precision, recall, f1, auc))

df_res = pd.DataFrame(results, columns=['threshold', 'precision', 'recall', 'f1', 'auc'])
print(df_res.to_string(index=False))

best = df_res[df_res['f1'] == df_res['f1'].max()].iloc[0]
print(f"\nBest threshold: {best.threshold:.2f} -> F1 = {best.f1:.3f}, Recall = {best.recall:.3f}")

# ---------- optional export ----------
if args.save:
    df_res.to_csv(args.save, index=False)
    print(f"Saved results to {args.save}")
if args.save_json:
    with open(args.save_json, 'w', encoding='utf-8') as f:
        json.dump({'threshold': float(best.threshold)}, f, indent=2)
    print(f"Saved best threshold to {args.save_json}")

# ---------- plot ----------
plt.figure(figsize=(10, 6))
plt.plot(df_res['threshold'], df_res['f1'], label='F1-score', marker='o')
plt.plot(df_res['threshold'], df_res['recall'], label='Recall', marker='x')
plt.plot(df_res['threshold'], df_res['precision'], label='Precision', marker='^')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Threshold Optimization Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

