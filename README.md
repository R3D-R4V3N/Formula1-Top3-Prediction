# F1-Top3-Prediction

This repository contains utilities to download Formula 1 race data using the Jolpica F1 API. The data can be used for analytics or building predictive models.

## Data Collection

Raw responses are downloaded with `fetch_data.py` and stored under `jolpica_f1_cache/<season>/<round>.json`. `process_data.py` reads the cache and builds `f1_data_2022_to_present.csv`. A helper script `data_collection.py` runs both steps. The processed data includes:

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
- relative qualifying time delta in seconds
- relative qualifying time delta as a percentage of pole time
- teammate qualifying gap in seconds
- driver momentum over the last three races
- pit stop difficulty index

The script writes the prepared dataset to `f1_data_2022_to_present.csv` in the current directory.

### Requirements

- Python 3.8+
- `requests` library

Install the dependency with:

```bash
pip install requests
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
