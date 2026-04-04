import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(source: str = "csv") -> pd.DataFrame:
    """
    Load Netflix (NFLX) stock data.
    source: "csv"  — load from data/netflix.csv (default, works offline)
            "live" — pull latest data from Yahoo Finance via yfinance
    """
    if source == "live":
        return _load_live()
    return _load_csv()


def _load_csv() -> pd.DataFrame:
    logger.info("Loading data from data/netflix.csv")
    return pd.read_csv("data/netflix.csv", sep="\t")


def _load_live() -> pd.DataFrame:
    try:
        import yfinance as yf
        logger.info("Fetching live NFLX data from Yahoo Finance...")
        ticker = yf.Ticker("NFLX")
        df = ticker.history(period="max")
        df = df.reset_index()
        df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                                  "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df["Stock Splits"] = 0
        df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
        logger.info(f"Fetched {len(df):,} rows from Yahoo Finance")
        return df
    except ImportError:
        logger.warning("yfinance not installed, falling back to CSV")
        return _load_csv()
    except Exception as e:
        logger.warning(f"Live fetch failed ({e}), falling back to CSV")
        return _load_csv()
