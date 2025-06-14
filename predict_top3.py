import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from fetch_data import (
    get_qualifying_results,
    get_driver_standings,
    get_constructor_standings,
    get_round_info,
)
from process_data import parse_qual_time
from model_catboost_final import MODEL_PARAMS, THRESHOLD


def compute_momentum(history):
    if len(history) >= 7:
        last3 = history[-1] - history[-4]
        prev3 = history[-4] - history[-7]
        return last3 - prev3
    return 0.0


def build_features(season: int, round_no: int, hist_df: pd.DataFrame) -> pd.DataFrame:
    race = get_round_info(season, round_no)
    circuit_id = race.get("Circuit", {}).get("circuitId")

    qual_results = get_qualifying_results(season, round_no)

    # Only rely on qualifying data to build the driver list
    results = []
    for qr in qual_results:
        results.append(
            {
                "Driver": qr.get("Driver", {}),
                "Constructor": qr.get("Constructor", {}),
                "grid": qr.get("position"),  # use qualifying position as grid
            }
        )

    best_times = {}
    qual_pos = {}
    for qr in qual_results:
        drv = qr["Driver"]["driverId"]
        times = [parse_qual_time(qr.get(q)) for q in ("Q1", "Q2", "Q3")]
        times = [t for t in times if t is not None]
        best_times[drv] = min(times) if times else None
        try:
            pos = int(qr.get("position"))
        except (TypeError, ValueError):
            pos = None
        qual_pos[drv] = pos

    pole_time = None
    valid = [t for t in best_times.values() if t is not None]
    if valid:
        pole_time = min(valid)

    ds_prev = {
        d["Driver"]["driverId"]: d for d in get_driver_standings(season, round_no - 1)
    }
    cs_prev = {
        c["Constructor"]["constructorId"]: c
        for c in get_constructor_standings(season, round_no - 1)
    }

    driver_hist = (
        hist_df.groupby("driver_id")["driver_points_scored"]
        .apply(lambda s: s.sort_index().tolist())
        .to_dict()
    )
    cons_hist = (
        hist_df.groupby("constructor_id")["constructor_points_scored"]
        .apply(lambda s: s.sort_index().tolist())
        .to_dict()
    )

    mean_psd = hist_df["pit_stop_difficulty"].mean()

    rows = []
    for res in results:
        drv = res["Driver"]["driverId"]
        constructor = res["Constructor"]["constructorId"]
        try:
            grid_pos = int(res.get("grid"))
        except (TypeError, ValueError):
            grid_pos = None

        q_pos = qual_pos.get(drv)

        # Grid penalties for the upcoming race are unknown at qualifying time
        penalty_places = 0
        penalty_flag = 0
        bonus_flag = 0

        q2_flag = 1 if q_pos is not None and q_pos <= 15 else 0
        q3_flag = 1 if q_pos is not None and q_pos <= 10 else 0

        best_time = best_times.get(drv)
        gap_sec = (
            best_time - pole_time
            if best_time is not None and pole_time is not None
            else 5.0
        )
        gap_pct = (
            (best_time / pole_time - 1) * 100
            if best_time is not None and pole_time is not None
            else 5.0
        )

        teammates = [
            r["Driver"]["driverId"]
            for r in results
            if r["Constructor"]["constructorId"] == constructor
            and r["Driver"]["driverId"] != drv
        ]
        teammate_best = min(
            [best_times.get(t) for t in teammates if best_times.get(t) is not None],
            default=None,
        )
        teammate_gap = (
            best_time - teammate_best
            if best_time is not None and teammate_best is not None
            else 5.0
        )

        ds = ds_prev.get(drv, {})
        cs = cs_prev.get(constructor, {})

        try:
            dr_rank = int(ds.get("position"))
        except (TypeError, ValueError):
            dr_rank = None
        driver_points = float(ds.get("points", 0.0))

        try:
            cons_rank = int(cs.get("position"))
        except (TypeError, ValueError):
            cons_rank = None
        cons_points = float(cs.get("points", 0.0))

        momentum = compute_momentum(driver_hist.get(drv, []))
        cons_momentum = compute_momentum(cons_hist.get(constructor, []))

        rows.append(
            dict(
                season_year=season,
                round_number=round_no,
                circuit_id=circuit_id,
                driver_id=drv,
                starting_grid_position=grid_pos,
                grid_penalty_places=penalty_places,
                grid_penalty_flag=penalty_flag,
                grid_bonus_flag=bonus_flag,
                q2_flag=q2_flag,
                q3_flag=q3_flag,
                driver_points_scored=driver_points,
                driver_championship_rank=dr_rank,
                constructor_id=constructor,
                constructor_points_scored=cons_points,
                constructor_championship_rank=cons_rank,
                rqtd_sec=gap_sec,
                rqtd_pct=gap_pct,
                teammate_quali_gap_sec=teammate_gap,
                driver_momentum=momentum,
                constructor_momentum=cons_momentum,
                pit_stop_difficulty=mean_psd,
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict F1 podium for a race")
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--round", type=int, required=True, help="Round number")
    args = parser.parse_args()

    csv_path = Path(__file__).with_name("f1_data_2022_to_present.csv")
    df = pd.read_csv(csv_path)
    df["top3_flag"] = (df["finishing_position"] <= 3).astype(int)
    df["group"] = df["season_year"].astype(str) + "-" + df["round_number"].astype(str)

    train_df = df[
        (df["season_year"] < args.season)
        | ((df["season_year"] == args.season) & (df["round_number"] < args.round))
    ]

    X = train_df.drop(columns=["finishing_position", "top3_flag", "group"])
    y = train_df["top3_flag"].values
    cat_cols = ["circuit_id", "driver_id", "constructor_id"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    params = MODEL_PARAMS.copy()
    params["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

    model = CatBoostClassifier(**params)
    train_pool = Pool(X, y, cat_features=cat_idx)
    model.fit(train_pool)

    features = build_features(args.season, args.round, train_df)
    preds = model.predict_proba(Pool(features, cat_features=cat_idx))[:, 1]
    features["prob"] = preds
    top3 = (
        features.sort_values("prob", ascending=False)
        .head(3)["driver_id"]
        .tolist()
    )

    output_csv = Path(__file__).with_name(
        f"prediction_data_{args.season}_{args.round}.csv"
    )
    features.to_csv(output_csv, index=False)

    print("Predicted podium drivers:")
    for drv in top3:
        print(drv)


if __name__ == "__main__":
    main()
