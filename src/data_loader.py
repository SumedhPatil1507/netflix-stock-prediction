from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(source: str = "csv") -> pd.DataFrame:
    """
    Load Netflix (NFLX) stock data.

    Parameters
    ----------
    source : "csv"  — load from data/netflix.csv (default, works offline)
             "live" — pull latest data from Yahoo Finance via yfinance
    """
    if source == "live":
        return _load_live()
    return _load_csv()


def _load_csv() -> pd.DataFrame:
    logger.info("Loading data from data/netflix.csv")
    df = pd.read_csv("data/netflix.csv", sep="\t")
    return _validate(df)


def _load_live() -> pd.DataFrame:
    try:
        import yfinance as yf
        logger.info("Fetching live NFLX data from Yahoo Finance...")
        ticker = yf.Ticker("NFLX")
        df = ticker.history(period="max")
        df = df.reset_index()
        if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                                  "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df["Stock Splits"] = 0
        df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
        logger.info(f"Fetched {len(df):,} rows from Yahoo Finance")
        return _validate(df)
    except ImportError:
        logger.warning("yfinance not installed, falling back to CSV")
        return _load_csv()
    except Exception as e:
        logger.warning(f"Live fetch failed ({e}), falling back to CSV")
        return _load_csv()


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic OHLCV sanity checks — logs warnings, does not raise."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return df

    n_before = len(df)
    # Drop rows where Close is NaN or zero
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    # Drop rows where High < Low (data error)
    df = df[df["High"] >= df["Low"]]
    # Drop rows where Volume is negative
    df = df[df["Volume"] >= 0]

    dropped = n_before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} invalid rows during validation")

    logger.info(f"Data validated: {len(df):,} rows")
    return df
