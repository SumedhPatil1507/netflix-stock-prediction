from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Default ticker — overridable via config or env
DEFAULT_TICKER = "NFLX"


def load_data(source: str = "csv", ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    """
    Load stock OHLCV data for any ticker.

    Parameters
    ----------
    source : "csv"  — load from data/{ticker}.csv (tab-separated)
             "live" — pull from Yahoo Finance via yfinance
    ticker : stock ticker symbol (default: NFLX)
    """
    if source == "live":
        return _load_live(ticker)
    return _load_csv(ticker)


def _load_csv(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    path = f"data/{ticker.upper()}.csv"
    # Fallback to legacy netflix.csv for backward compatibility
    import os
    if not os.path.exists(path):
        path = "data/netflix.csv"
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path, sep="\t")
    return _validate(df)


def _load_live(ticker: str = DEFAULT_TICKER) -> pd.DataFrame:
    try:
        import yfinance as yf
        logger.info(f"Fetching live {ticker} data from Yahoo Finance...")
        t  = yf.Ticker(ticker)
        df = t.history(period="max")
        df = df.reset_index()
        if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
            df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                                  "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df["Stock Splits"] = 0
        df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
        logger.info(f"Fetched {len(df):,} rows for {ticker}")
        return _validate(df)
    except ImportError:
        logger.warning("yfinance not installed, falling back to CSV")
        return _load_csv(ticker)
    except Exception as e:
        logger.warning(f"Live fetch failed ({e}), falling back to CSV")
        return _load_csv(ticker)


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic OHLCV sanity checks — logs warnings, does not raise."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return df

    n_before = len(df)
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    df = df[df["High"] >= df["Low"]]
    df = df[df["Volume"] >= 0]

    dropped = n_before - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} invalid rows during validation")

    logger.info(f"Data validated: {len(df):,} rows")
    return df
