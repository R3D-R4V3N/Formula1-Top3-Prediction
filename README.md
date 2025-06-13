# F1-Top3-Prediction

This repository contains utilities to download Formula 1 race data using the Jolpica F1 API. The data can be used for analytics or building predictive models.

## Data Collection

`data_collection.py` retrieves all race results for a range of seasons. The data
is written incrementally to `f1_results.csv` and includes at least the driver ID
and grid/finish positions for each race. A separate file `net_gain_5.csv`
contains the rolling five-race average of grid position minus finish position
for every driver.

### Requirements

- Python 3.8+
- `requests-cache`
- `pandas`

Install the dependencies with:

```bash
pip install requests-cache pandas
```

### Usage

Run the script from the command line specifying the start and end season:

```bash
python data_collection.py --start-season 2022 --end-season 2024
```

The downloader respects the Jolpica API quota of **4 requests per second** and
**500 requests per hour**. Because results are downloaded in bulk pages and are
cached locally under `f1_cache/`, only around 40 requests are required for an
entire season.

## API Endpoints

The API endpoints used by the script are documented in the `endpoints` directory. Results are retrieved via the season level endpoint shown below:

- `/{season}/results.json?limit=100&offset=n`

Refer to the markdown files in the `endpoints` folder for details about optional query parameters and example responses.
