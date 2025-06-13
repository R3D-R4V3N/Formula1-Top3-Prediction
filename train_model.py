import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


def load_data(path: str) -> pd.DataFrame:
    """Load the F1 dataset"""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple:
    """Prepare features and target"""
    df = df.copy()
    df['top3'] = (df['finishing_position'].astype(float) <= 3).astype(int)
    feature_cols = [
        'season_year',
        'round_number',
        'starting_grid_position',
        'grid_penalty_places',
        'grid_penalty_flag',
        'q2_flag',
        'q3_flag',
        'driver_points_scored',
        'driver_championship_rank',
        'constructor_points_scored',
        'constructor_championship_rank',
        'rqtd_sec',
        'rqtd_pct',
        'driver_id',
        'constructor_id',
        'circuit_id',
    ]
    df = df[feature_cols + ['top3']]
    df = pd.get_dummies(df, columns=['driver_id', 'constructor_id', 'circuit_id'])
    X = df.drop('top3', axis=1)
    y = df['top3']
    return X, y


def train_model(X, y, model_name: str = "xgb"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_name == "xgb":
        model = XGBClassifier(
            n_estimators=5000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    elif model_name == "catboost":
        model = CatBoostClassifier(
            iterations=5000,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=100,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    elif model_name == "lgbm":
        model = LGBMClassifier(
            n_estimators=5000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    preds = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
    }
    return model, metrics


def save_metrics(metrics: dict, path: str) -> None:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Train F1 top3 prediction model')
    parser.add_argument('--data', default='f1_data_2022_to_present.csv', help='CSV data path')
    parser.add_argument('--metrics', default='training_metrics.csv', help='Path to write metrics CSV')
    parser.add_argument('--model', default='xgb', choices=['xgb', 'catboost', 'lgbm'], help='Model type to train')
    args = parser.parse_args()

    df = load_data(args.data)
    X, y = preprocess(df)
    model, metrics = train_model(X, y, args.model)
    save_metrics(metrics, args.metrics)


if __name__ == '__main__':
    main()
