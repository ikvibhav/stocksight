from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yfinance as yf


class StockReader(ABC):

    @abstractmethod
    def get_web_stock_data_period(
        self,
        stock_symbol: str,
        period: str,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_web_stock_data(
        self,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_stock_data(
        self,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
        file_path: str,
    ) -> None:
        pass

    @abstractmethod
    def read_local_stock_data(self, file_path: str):
        pass

    def read_csv(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


class Snp500Reader(StockReader):

    def get_web_stock_data_period(self, stock_symbol: str, period: str) -> pd.DataFrame:
        """
        Get stock data from Yahoo Finance API for a given stock symbol and period

        Args:
            stock_symbol: str: The stock symbol to get data for
            period: str: The period to get data for

        Returns:
            pd.DataFrame: The stock data
        """
        download_df = yf.download(tickers=stock_symbol, period=period)

        if isinstance(download_df.columns, pd.MultiIndex):
            download_df.columns = download_df.columns.droplevel(1)

        return download_df

    def get_web_stock_data(
        self, stock_symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get stock data from Yahoo Finance API for a given stock symbol and date range

        Args:
            stock_symbol: str: The stock symbol to get data for
            start_date: datetime: The start date to get data for
            end_date: datetime: The end date to get data for

        Returns:
            pd.DataFrame: The stock data
        """

        download_df = yf.download(stock_symbol, start=start_date, end=end_date)

        if isinstance(download_df.columns, pd.MultiIndex):
            download_df.columns = download_df.columns.droplevel(1)

        return download_df

    def save_stock_data(
        self,
        stock_symbol: str,
        start_date: datetime,
        end_date: datetime,
        output_file_path: str,
    ) -> None:
        """
        Save stock data from Yahoo Finance API for a given stock symbol and date range to a CSV file

        Args:
            stock_symbol: str: The stock symbol to get data for
            start_date: datetime: The start date to get data for
            end_date: datetime: The end date to get data for
            output_file_path: str: The path to save the CSV file to

        Returns:
            None
        """
        data = self.get_web_stock_data(stock_symbol, start_date, end_date)
        data.to_csv(output_file_path)

    def read_local_stock_data(self, file_path: str) -> pd.DataFrame:
        """
        Read stock data from a CSV file

        Args:
            file_path: str: The path to the CSV file

        Returns:
            pd.DataFrame: The stock data
        """
        return self.read_csv(file_path)


if __name__ == "__main__":
    stock_object = Snp500Reader()
    data = stock_object.get_web_stock_data_period("AAPL", "1mo")
    print(data)
