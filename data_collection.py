from datetime import datetime

from fetch_data import fetch_data
from process_data import prepare_dataset



if __name__ == "__main__":
    current_year = datetime.now().year
    fetch_data(2022, current_year)
    prepare_dataset(2022, current_year, "f1_data_2022_to_present.csv")
