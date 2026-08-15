import pandas as pd
import yfinance as yf


def yearly_percentage_change(stock_list: list) -> pd.DataFrame:
    yearly_changes = pd.DataFrame()

    for stock in stock_list:
        # Create a Ticker instance for the stock
        stock_ticker = yf.Ticker(stock)
        # Fetch maximum available historical data
        stock_hist = stock_ticker.history(period="max")

        # Calculate yearly percentage change
        stock_yearly_change = stock_hist["Close"].resample("YE").ffill().pct_change()

        # Convert the percentage change to a percentage
        stock_yearly_change = stock_yearly_change * 100

        # Convert the percentage to integer and ignore the NaN values
        stock_yearly_change = stock_yearly_change.round(0).dropna()

        # Misc Formatting
        stock_yearly_change.index = stock_yearly_change.index.strftime("%Y-%m-%d")

        # Add the results to the DataFrame
        yearly_changes[stock] = stock_yearly_change

    return yearly_changes


def correlation_analysis(tickers: str, period: str) -> pd.DataFrame:
    """
    Get the correlation between the adjusted close prices of the given stock symbols

    Args:
        tickers: str: The stock symbols to get data for
        period: str: The period to get data for

    Returns:
        pd.DataFrame: The correlation between the adjusted close prices of the given stocks
    """
    data = yf.download(tickers, period=period)
    return data["Close"].corr()
