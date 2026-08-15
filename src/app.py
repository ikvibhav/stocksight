import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import style

from common.utils import yearly_percentage_change
from common.base_stock_reader import Snp500Reader
from common.utils import correlation_analysis
from features.compute import (
    compute_atr,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
)

style.use("ggplot")


@st.cache_data
@st.cache_resource
def fetch_stock_data_period(stock_symbol, period):
    stock_object = Snp500Reader()
    return stock_object.get_web_stock_data_period(stock_symbol, period)


def render_time_series(data, selected_stock, interactive_chart):
    st.subheader(f"Stock Data for {selected_stock} for last 5 days")
    st.write(data.tail())

    data["50ma"] = data["Close"].rolling(window=50).mean()
    data["200ma"] = data["Close"].rolling(window=200).mean()

    st.subheader(f"Time Series Plot for {selected_stock}")
    st.write(
        "Golden Cross - 50 day moving average (short term) crosses above the 200 day moving average (long term). This is a bullish signal."
    )
    st.write(
        "Death Cross - 50 day moving average (short term) crosses below the 200 day moving average (long term)"
    )

    if interactive_chart:
        st.line_chart(data[["Close", "50ma", "200ma"]])
    else:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [5, 1]}
        )
        ax1.plot(data.index, data["Close"], label="Close Price")
        ax1.plot(data.index, data["50ma"], label="50 day moving average")
        ax1.plot(data.index, data["200ma"], label="200 day moving average")
        ax1.legend()
        ax1.set_ylabel("Price")
        ax2.bar(data.index, data["Volume"], label="Volume")
        ax2.set_ylabel("Volume")
        ax2.set_xlabel("Date")
        st.pyplot(fig)


def render_correlation_analysis(stock_list):
    period_corr = st.sidebar.radio(
        "Select the period", ["1mo", "3mo", "6mo", "1y", "2y"]
    )
    st.subheader("Correlation Analysis")
    st.write(f"Stocks - {stock_list}, Period - {period_corr}")
    df_corr = correlation_analysis(stock_list, period_corr)
    fig, ax = plt.subplots()
    sns.heatmap(df_corr, cmap="coolwarm", annot=True)
    ax.set_title(f"Correlation Matrix ({period_corr})")
    st.pyplot(fig)


def render_yearly_pct_changes(stock_list):
    period_years = st.sidebar.radio("Select the period (years)", ["5", "10", "20"])
    st.subheader("Yearly Percentage Changes")
    st.write(f"Selected Period - {period_years} years")
    df_pct = yearly_percentage_change(stock_list)
    st.write(df_pct.tail(int(period_years)))


def render_rsi_tab(data, selected_stock):
    rsi_df = compute_rsi(data, window=14)
    st.write("Relative Strength Index (RSI)")
    st.caption(
        "0-100 momentum scale. Above 70 can indicate overbought; below 30 can indicate oversold."
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(rsi_df.index, rsi_df["Close"], color="tab:blue", label="Close Price")
    ax1.set_ylabel("Close Price ($)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(rsi_df.index, rsi_df["rsi_14"], color="tab:red", label="RSI (14)")
    ax2.axhline(70, color="tab:red", linestyle="--", alpha=0.6)
    ax2.axhline(30, color="tab:orange", linestyle="--", alpha=0.6)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(f"{selected_stock} Price vs RSI")
    st.pyplot(fig)
    st.write(rsi_df[["Close", "rsi_14"]].tail())


def render_macd_tab(data):
    macd_df = compute_macd(data, fast=12, slow=26, signal=9)
    st.write("MACD")
    st.caption(
        "Trend/momentum view. Histogram above 0 suggests bullish momentum; below 0 suggests bearish momentum."
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    ax1.plot(macd_df.index, macd_df["Close"], color="tab:blue", label="Close Price")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")

    ax2.plot(macd_df.index, macd_df["macd_line"], color="tab:green", label="MACD Line")
    ax2.plot(
        macd_df.index, macd_df["macd_signal"], color="tab:orange", label="Signal Line"
    )
    ax2.bar(
        macd_df.index,
        macd_df["macd_histogram"],
        color="tab:gray",
        alpha=0.4,
        label="Histogram",
    )
    ax2.axhline(0, color="black", linewidth=1, alpha=0.6)
    ax2.set_ylabel("MACD")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    st.pyplot(fig)
    st.write(macd_df[["macd_line", "macd_signal", "macd_histogram"]].tail())


def render_bollinger_bands_tab(data):
    bb_df = compute_bollinger_bands(data, window=20, num_std=2.0)
    st.write("Bollinger Bands")
    st.caption(
        "Middle band is rolling average. Upper/lower bands expand when volatility rises."
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    ax1.plot(bb_df.index, bb_df["Close"], color="tab:blue", label="Close")
    ax1.plot(bb_df.index, bb_df["bb_middle"], color="tab:green", label="Middle (20)")
    ax1.plot(
        bb_df.index, bb_df["bb_upper"], color="tab:red", linestyle="--", label="Upper"
    )
    ax1.plot(
        bb_df.index,
        bb_df["bb_lower"],
        color="tab:orange",
        linestyle="--",
        label="Lower",
    )
    ax1.fill_between(
        bb_df.index, bb_df["bb_lower"], bb_df["bb_upper"], color="tab:gray", alpha=0.15
    )
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")

    ax2.plot(bb_df.index, bb_df["bb_bandwidth"], color="tab:purple", label="Bandwidth")
    ax2.set_ylabel("Bandwidth")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    st.pyplot(fig)
    st.write(bb_df[["bb_middle", "bb_upper", "bb_lower", "bb_bandwidth"]].tail())


def render_atr_tab(data):
    atr_df = compute_atr(data, window=14)
    st.write("Average True Range (ATR)")
    st.caption("Volatility measure. Higher ATR means larger recent price swings.")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    ax1.plot(atr_df.index, atr_df["Close"], color="tab:blue", label="Close Price")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")

    ax2.plot(atr_df.index, atr_df["atr_14"], color="tab:red", label="ATR (14)")
    ax2.set_ylabel("ATR")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")

    st.pyplot(fig)
    st.write(atr_df[["true_range", "atr_14"]].tail())


def render_technical_analysis(data, selected_stock):
    st.subheader("Technical Analysis")
    st.write(f"Selected Stock - {selected_stock}")

    tab_rsi, tab_macd, tab_bb, tab_atr = st.tabs(
        ["RSI", "MACD", "Bollinger Bands", "ATR"]
    )

    with tab_rsi:
        render_rsi_tab(data, selected_stock)
    with tab_macd:
        render_macd_tab(data)
    with tab_bb:
        render_bollinger_bands_tab(data)
    with tab_atr:
        render_atr_tab(data)


def render_sidebar():
    stock_list = ["^GSPC", "AMZN", "TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"]

    st.sidebar.title("1. Time Series Analysis")
    interactive_chart = st.sidebar.checkbox("Interactive Chart", value=False)
    selected_stock = st.sidebar.selectbox("Select a stock", stock_list)
    period = st.sidebar.radio(
        "Select the period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    )

    st.sidebar.title("2. Correlation Analysis")
    correlanalysis = st.sidebar.checkbox("Correlation Analysis", value=False)

    st.sidebar.title("3. Yearly Percentage Changes")
    yearlypct = st.sidebar.checkbox("Yearly Percentage Changes", value=False)

    st.sidebar.title("4. Technical Analysis")
    tech_analysis = st.sidebar.checkbox("Technical Analysis", value=False)

    return (
        stock_list,
        selected_stock,
        period,
        interactive_chart,
        correlanalysis,
        yearlypct,
        tech_analysis,
    )


def main():
    st.title("Stock Monitor")
    st.write("Welcome to the Stock Monitor App!")
    st.write(
        "This app will allow you to track the stock prices of your favorite companies."
    )
    st.write("Please select the stock you would like to track from the sidebar.")

    (
        stock_list,
        selected_stock,
        period,
        interactive_chart,
        correlanalysis,
        yearlypct,
        tech_analysis,
    ) = render_sidebar()

    data = fetch_stock_data_period(selected_stock, period)

    if data is not None and not data.empty:
        render_time_series(data, selected_stock, interactive_chart)
    else:
        st.write("No data available for the selected stock and date range.")

    if correlanalysis:
        render_correlation_analysis(stock_list)

    if yearlypct:
        render_yearly_pct_changes(stock_list)

    if tech_analysis:
        render_technical_analysis(data, selected_stock)


if __name__ == "__main__":
    main()
