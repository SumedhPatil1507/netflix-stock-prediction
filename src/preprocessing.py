from __future__ import annotations
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse dates, set as index, sort chronologically, remove duplicates.

    Parameters
    ----------
    df : raw OHLCV DataFrame with a 'Date' column

    Returns
    -------
    DataFrame with DatetimeIndex, sorted ascending, no duplicate dates
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df
