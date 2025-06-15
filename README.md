# F1-Top3-Prediction

This repository contains utilities to download Formula 1 race data using the Jolpica F1 API. The data can be used for analytics or building predictive models.

## Data Collection

 Raw responses and weather information are downloaded with `fetch_data.py` and stored under `jolpica_f1_cache/<season>/<round>.json` and `weather_cache/`. `process_data.py` reads the cached files and builds `f1_data_2022_to_present.csv`. A helper script `data_collection.py` runs both steps. The processed data includes:

- circuit ID
- start position on the grid
- finish position
- grid penalty places with penalty and bonus flags
- Q2 qualifier flag
- Q3 qualifier flag
- total driver championship points after each race
- driver championship position after each race
- team (constructor) championship points after each race
- team championship position after each race
- relative qualifying time delta in seconds (uses 5.0 if no valid time)
- relative qualifying time delta as a percentage of pole time (uses 5.0 if no valid time)
- teammate qualifying gap in seconds (uses 5.0 if no valid time for either driver)
- driver momentum over the last three races (0.0 for the first six rounds)
- constructor momentum over the last three races (0.0 for the first six rounds)
- pit stop difficulty index
- mean temperature during the race window
- total precipitation during the race window
- mean humidity during the race window
- mean wind speed during the race window

The script writes the prepared dataset to `f1_data_2022_to_present.csv` in the current directory.

### Requirements

- Python 3.8+
- `requests` library
- `meteostat`
- `pyowm` (optional, for weather forecasts)

Install the dependencies with:

```bash
pip install requests meteostat pyowm
```

### Usage

First fetch the raw data:

```bash
python fetch_data.py
```

Then generate the CSV using the cached responses:

```bash
python process_data.py
```

Alternatively run `python data_collection.py` to execute both steps in one go.

The Jolpica API enforces a rate limit of **4 requests per second** and **500 requests per hour**. `data_collection.py` includes a simple rate limiter and will pause if a `429` error is returned. It can be executed multiple times; when the CSV already exists, the script resumes from the last recorded race and appends new data. While running, the script prints log messages with a few emojis so you can easily follow its progress in your terminal.

## API Endpoints

All endpoints used by the script are documented in the `endpoints` directory. They follow the structure shown below:

- `/{season}/{round}/results.json`
- `/{season}/{round}/driverStandings.json`
- `/{season}/{round}/constructorStandings.json`
- `/{season}/{round}/pitstops/`

Refer to the markdown files in the `endpoints` folder for details about optional query parameters and example responses.

## Podium Prediction

Use `predict_top3.py` to predict the top three finishers for a specific race. The
script trains only on races completed prior to the chosen round and relies solely
on information available up to qualifying of that event.

```bash
python predict_top3.py --season 2025 --round 9
```

The command above predicts the Spanish Grand Prix (round 9) for the 2025 season.

## Model Evaluation

`model_catboost_final.py` now uses a time-series split that trains on past races
and tests on later events. Running the script prints cross-validation metrics
similar to:

```
[Fold 1] acc=0.883  prec=1.000  rec=0.222  f1=0.364  auc=0.956
[Fold 2] acc=0.917  prec=0.690  rec=0.806  f1=0.744  auc=0.946
[Fold 3] acc=0.933  prec=0.700  rec=0.972  f1=0.814  auc=0.980
[Fold 4] acc=0.925  prec=0.725  rec=0.806  f1=0.763  auc=0.956
[Fold 5] acc=0.916  prec=0.654  rec=0.944  f1=0.773  auc=0.967

=== Tuned CatBoost Results (mean ± std) ===
acc: 0.915 ± 0.017
prec: 0.754 ± 0.125
rec: 0.750 ± 0.273
f1: 0.691 ± 0.165
auc: 0.961 ± 0.012
```

## Hyper-parameter Tuning

`tune_catboost_optuna_cpu.py` and `tune_catboost_optuna_gpu.py` search for the
best CatBoost parameters using a time-series split to respect race order. After
updating `f1_data_2022_to_present.csv`, re-run one of these scripts to obtain
fresh parameters before training the final model. `threshold_scan_final.py`
uses the same time-series folds when scanning decision thresholds.
