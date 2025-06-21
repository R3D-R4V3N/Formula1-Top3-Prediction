import numpy as np
from pathlib import Path
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from group_time_series_split import GroupTimeSeriesSplit


def load_or_create_calibrator(
    X,
    y,
    groups,
    cat_idx,
    params,
    path: Path = Path("calibrator.joblib"),
):
    """Return a logistic regression calibrator fitted on OOF probabilities."""
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    oof = np.zeros_like(y, dtype=float)
    n_groups = len(np.unique(groups))
    n_splits = min(5, max(1, n_groups - 1))
    if n_splits > 1:
        tscv = GroupTimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(X, groups=groups):
            model = CatBoostClassifier(**params)
            tr_pool = Pool(X.iloc[train_idx], y[train_idx], cat_features=cat_idx)
            val_pool = Pool(X.iloc[test_idx], y[test_idx], cat_features=cat_idx)
            model.fit(tr_pool, eval_set=val_pool, early_stopping_rounds=300, verbose=False)
            oof[test_idx] = model.predict_proba(Pool(X.iloc[test_idx], cat_features=cat_idx))[:, 1]
    else:
        model = CatBoostClassifier(**params)
        model.fit(Pool(X, y, cat_features=cat_idx), verbose=False)
        oof = model.predict_proba(Pool(X, cat_features=cat_idx))[:, 1]
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(oof.reshape(-1, 1), y)
    joblib.dump(calibrator, path)
    return calibrator
