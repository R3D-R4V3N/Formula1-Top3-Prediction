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
- circuit podium rate
- constructor podium rate
- driver DNF rate (rolling window)
- constructor DNF rate (rolling window)
- pit stop difficulty index
- mean temperature forecast for the race window
- total precipitation forecast for the race window
- mean humidity forecast for the race window
- mean wind speed forecast for the race window

The weather metrics use the forecast that would have been available at
qualifying time. For completed races this is approximated using a
10-year historical average for the same location and time window.

The script writes the prepared dataset to `f1_data_2022_to_present.csv` in the current directory.

### Requirements

- Python 3.8+
- `requests` library
- `meteostat` (tested with >=1.7; older versions used a deprecated data endpoint)
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

### Streamlit Web App

You can also explore predictions interactively with Streamlit:

```bash
streamlit run streamlit_app.py
```

The app lets you choose a season and round, shows the predicted probabilities
for each driver, and visualizes global and per-driver feature importance using
SHAP values.

## Advanced Model Tuning

The repository includes utilities to tune the CatBoost model and optimise the
decision threshold. A typical workflow is:

1. Fetch and process the latest race data:

   ```bash
   python data_collection.py
   ```

2. Run Optuna hyper-parameter search (use `--gpu` on systems with a GPU):

   ```bash
   python tune_catboost_optuna_cpu.py --trials 300 --output optuna_best_params.json
   ```

   The script writes the best model parameters and a suggested threshold to
   `optuna_best_params.json`.
   Older files using keys like `lr` or `l2` are automatically converted.

3. Determine the final decision threshold with out-of-fold predictions:

   ```bash
   python threshold_scan_final.py --calibrate \
       --params optuna_best_params.json \
       --save threshold_results.csv \
       --save-json best_threshold.json
   ```

   This saves the threshold sweep to `threshold_results.csv` and the best value
   to `best_threshold.json`.

4. Train and evaluate the final calibrated model:

   ```bash
   python model_catboost_final.py
   ```

The script automatically loads the parameters and threshold from the JSON files
if they are present.
