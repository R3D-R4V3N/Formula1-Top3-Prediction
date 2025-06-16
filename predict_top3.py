import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from fetch_data import (
    get_qualifying_results,
    get_driver_standings,
    get_constructor_standings,
    get_round_info,
    fetch_weather,
)
from process_data import parse_qual_time

DNF_WINDOW = 5


def is_dnf(status: str) -> bool:
    if not status:
        return True
    status = status.lower()
    return not ("finished" in status or "lap" in status)
from model_catboost_final import MODEL_PARAMS, THRESHOLD


def compute_momentum(history):
    if len(history) >= 6:
        last3 = history[-1] - history[-4]
        prev3 = history[-4] - history[-6]
        return last3 - prev3
    return 0.0


def compute_last3_performance(history):
    if len(history) >= 4:
        return (history[-1] - history[-4]) / 3
    elif history:
        return (history[-1] - history[0]) / len(history)
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

    if round_no > 1:
        ds_prev = {
            d["Driver"]["driverId"]: d
            for d in get_driver_standings(season, round_no - 1)
        }
        cs_prev = {
            c["Constructor"]["constructorId"]: c
            for c in get_constructor_standings(season, round_no - 1)
        }
    else:
        ds_prev = {}
        cs_prev = {}

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

    if "dnf_flag" in hist_df.columns:
        driver_dnf_hist = (
            hist_df.groupby("driver_id")["dnf_flag"]
            .apply(lambda s: s.sort_index().tolist())
            .to_dict()
        )
        cons_dnf_hist = (
            hist_df.groupby("constructor_id")["dnf_flag"]
            .apply(lambda s: s.sort_index().tolist())
            .to_dict()
        )
    else:
        # If DNF information is unavailable, fall back to empty histories
        driver_dnf_hist = {d: [] for d in hist_df["driver_id"].unique()}
        cons_dnf_hist = {c: [] for c in hist_df["constructor_id"].unique()}

    circuit_stats = (
        hist_df.groupby("circuit_id")["top3_flag"]
        .agg(["sum", "count"])
        .to_dict("index")
    )
    constructor_stats = (
        hist_df.groupby("constructor_id")["top3_flag"]
        .agg(["sum", "count"])
        .to_dict("index")
    )

    past_psd = hist_df.loc[
        hist_df["circuit_id"] == circuit_id, "pit_stop_difficulty"
    ]
    mean_psd = past_psd.mean()
    weather = fetch_weather(season, round_no)

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

        drv_hist = driver_hist.get(drv, [])
        cons_hist_list = cons_hist.get(constructor, [])

        last3_perf = compute_last3_performance(drv_hist)
        cons_last3_perf = compute_last3_performance(cons_hist_list)

        momentum = compute_momentum(drv_hist)
        cons_momentum = compute_momentum(cons_hist_list)

        circ_stat = circuit_stats.get(circuit_id, {"sum": 0, "count": 0})
        circuit_podium_rate = (
            circ_stat["sum"] / circ_stat["count"] if circ_stat["count"] else 0.0
        )
        cons_stat = constructor_stats.get(constructor, {"sum": 0, "count": 0})
        constructor_podium_rate = (
            cons_stat["sum"] / cons_stat["count"] if cons_stat["count"] else 0.0
        )

        d_hist = driver_dnf_hist.get(drv, [])
        c_hist = cons_dnf_hist.get(constructor, [])
        driver_dnf_rate = (
            sum(d_hist[-DNF_WINDOW:]) / len(d_hist[-DNF_WINDOW:]) if d_hist else 0.0
        )
        constructor_dnf_rate = (
            sum(c_hist[-DNF_WINDOW:]) / len(c_hist[-DNF_WINDOW:]) if c_hist else 0.0
        )

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
                driver_last3_performance=last3_perf,
                driver_momentum=momentum,
                constructor_last3_performance=cons_last3_perf,
                constructor_momentum=cons_momentum,
                circuit_podium_rate=circuit_podium_rate,
                constructor_podium_rate=constructor_podium_rate,
                driver_dnf_rate=driver_dnf_rate,
                constructor_dnf_rate=constructor_dnf_rate,
                pit_stop_difficulty=mean_psd,
                temp_mean=weather.get("temp_mean"),
                precip_sum=weather.get("precip_sum"),
                humidity_mean=weather.get("humidity_mean"),
                wind_mean=weather.get("wind_mean"),
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

    drop_cols = ["finishing_position", "top3_flag", "group"]
    if "dnf_flag" in train_df.columns:
        drop_cols.append("dnf_flag")

    X = train_df.drop(columns=drop_cols)
    y = train_df["top3_flag"].values
    cat_cols = ["circuit_id", "driver_id", "constructor_id"]
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    params = MODEL_PARAMS.copy()
    params["class_weights"] = [1.0, (y == 0).sum() / (y == 1).sum()]

    model = CatBoostClassifier(**params)
    train_pool = Pool(X, y, cat_features=cat_idx)
    model.fit(train_pool)

    features = build_features(args.season, args.round, train_df)
    # Ensure prediction features align with training column order
    features = features[X.columns]
    preds = model.predict_proba(Pool(features, cat_features=cat_idx))[:, 1]
    features["prob"] = preds
    top3 = features.sort_values("prob", ascending=False).head(3)["driver_id"].tolist()

    print("Predicted podium drivers:")
    for drv in top3:
        print(drv)


if __name__ == "__main__":
    main()
