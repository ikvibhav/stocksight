from pathlib import Path

import pandas as pd
import yfinance as yf
from prefect import flow, task

EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FOLDER_PATH = Path("data/raw/")
FOLDER_PATH.mkdir(parents=True, exist_ok=True)

REFERENCE_PATH = Path("data/reference/")
REFERENCE_PATH.mkdir(parents=True, exist_ok=True)


@task
def save_reference_snapshot(data: pd.DataFrame, ticker: list[str], period: str) -> None:
    """
    Save a reference snapshot of the fetched data for drift detection and monitoring purposes.
    Implements: FR-DI-004

    Args:
        data (pd.DataFrame): The DataFrame containing the historical stock data.
        ticker (list[str]): The list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): The period for which to fetch data (e.g., '1y', '5d').
    """
    existing = list(REFERENCE_PATH.glob("*.csv"))
    if existing:
        print("Reference snapshot already exists. Skipping.")
        return
    filename = "".join(ticker) + f"_{period}_reference.csv"
    data.to_csv(REFERENCE_PATH / filename, index=True)
    print(f"Reference snapshot saved to {REFERENCE_PATH / filename}.")


def build_filename(ticker: list[str], period: str) -> str:
    """
    Build a filename for the CSV file based on the ticker, period, and current timestamp.
    Implements: FR-DI-003

    Args:
        ticker (list[str]): The list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): The period for which to fetch data (e.g., '1y', '5d').
    Returns:
        str: The generated filename for the CSV file.
    """
    iso_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    return "".join(ticker) + f"_{period}_{iso_timestamp}.csv"


@task
def fetch_data(ticker: list[str], period: str) -> pd.DataFrame:
    """
    Fetch historical stock data for a given ticker and period.
    Implements: FR-DI-001

    Args:
        ticker (list[str]): The list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): The period for which to fetch data (e.g., '1y', '5d').

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """

    # Fetch the historical data
    data = yf.download(ticker, period=period)

    return data


@task
def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate the fetched data to ensure it is not empty and contains the expected columns.
    Implements: FR-DI-002

    Args:
        data (pd.DataFrame): The DataFrame containing the historical stock data.

    Returns:
        bool: True if the data is valid, False otherwise.
    """
    # 1. Check if the DataFrame is empty
    if data.empty:
        print("Data validation failed: DataFrame is empty.")
        return False

    # 2. Check if the expected columns are present
    if isinstance(data.columns, pd.MultiIndex):
        columns = set(data.columns.get_level_values(0))
    else:
        columns = set(data.columns)
    if not all(column in columns for column in EXPECTED_COLUMNS):
        print(
            f"Data validation failed: Missing expected columns. Expected columns are {EXPECTED_COLUMNS}, but got {columns}."
        )
        return False

    # 3. Check for null values in the expected columns
    # If today's or yesterday's data is missing some values (e.g., due to market hours)
    # we can allow nulls in the last row but not in the historical data
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    yesterday_str = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if data.index[-1].strftime("%Y-%m-%d") == today_str or data.index[-1].strftime("%Y-%m-%d") == yesterday_str:
        print("Latest data point is from today or yesterday, allowing nulls in the last row if present.")
        past_data = data.iloc[:-1]
        if past_data[EXPECTED_COLUMNS].isnull().any().any():
            print("Data validation failed: Null values found in the expected columns of historical data.")
            return False
    else:
        if data[EXPECTED_COLUMNS].isnull().any().any():
            print("Data validation failed: Null values found in the expected columns.")
            return False

    print("Data validation successful.")
    return True


@task
def save_data(data: pd.DataFrame, filename: str) -> None:
    """
    Save the validated data to a CSV file

    Args:
        data (pd.DataFrame): The DataFrame containing the historical stock data.
        filename (str): The name of the file to save the data to.
    """
    # Save the DataFrame to a CSV file
    data.to_csv(FOLDER_PATH / filename, index=True)
    print(f"Data saved successfully to {FOLDER_PATH / filename}.")


@flow
def data_ingestion_pipeline(ticker: list[str], period: str, save_to_file: bool) -> None:
    """
    The main flow for the data ingestion pipeline.

    Args:
        ticker (list[str]): The list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        period (str): The period for which to fetch data (e.g., '1y', '5d').
        save_to_file (bool): Whether to save the fetched data to a file.
    """
    data = fetch_data(ticker, period)
    if validate_data(data):
        save_reference_snapshot(data, ticker, period)
        if save_to_file:
            save_data(data, build_filename(ticker, period))


if __name__ == "__main__":
    # Can also be called manually and automated from prefect UI
    # Implements: FR-DI-005
    data_ingestion_pipeline(ticker=["AAPL", "MSFT"], period="1y", save_to_file=True)
