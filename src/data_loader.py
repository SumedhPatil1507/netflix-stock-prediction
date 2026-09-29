"""
Multi-source data ingestion layer.
Supports:
  - TimescaleDB (hardened real-time persistence)
  - Alpha Vantage (daily + intraday 1min/5min, free tier)
  - Alpaca Markets (minute bars, free paper trading account)

Set API keys in .env:
  ALPHA_VANTAGE_KEY=your_key
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
  ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""
from __future__ import annotations
import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Literal
from src.time_series_db import save_ohlcv_data, load_ohlcv_data

logger = logging.getLogger(__name__)

DEFAULT_TICKER = os.getenv("DEFAULT_TICKER", "NFLX")

# Supported intervals for intraday
Interval = Literal["1min", "5min", "15min", "30min", "60min", "daily"]

def load_data(
    source: str = "database",
    ticker: str = DEFAULT_TICKER,
    interval: Interval = "daily",
    days_back: int = 365,
) -> pd.DataFrame:
    """
    Unified data loader supporting database lookup and API bootstrap.

    Parameters
    ----------
    source   : "database" | "alphavantage" | "alpaca"
    ticker   : stock symbol
    interval : data frequency (daily or intraday)
    days_back: how many calendar days of history to fetch
    """
    source = source.lower()
    ticker = ticker.upper()

    # Treat old csv / yfinance / live source requests as database requests
    if source in ("database", "csv", "yfinance", "live"):
        df = load_ohlcv_data(ticker, days_back)
        if not df.empty:
            logger.info(f"Loaded {len(df)} rows from TimescaleDB for {ticker}")
            return df
        
        # If DB has no data, bootstrap using Alpaca
        logger.info(f"TimescaleDB storage empty for {ticker}. Bootstrapping from Alpaca API...")
        source = "alpaca"

    if source == "alphavantage":
        df = _load_alpha_vantage(ticker, interval)
    elif source == "alpaca":
        df = _load_alpaca(ticker, interval, days_back)
    else:
        df = load_ohlcv_data(ticker, days_back)

    if not df.empty:
        # Cache raw data in TimescaleDB for future queries
        save_ohlcv_data(df, ticker)
        
    return df

# ── Alpha Vantage ─────────────────────────────────────────────────────────────
def _load_alpha_vantage(ticker: str, interval: Interval = "daily") -> pd.DataFrame:
    """
    Alpha Vantage REST API.
    Free tier: 25 requests/day.
    """
    api_key = os.getenv("ALPHA_VANTAGE_KEY")
    if not api_key:
        logger.warning("ALPHA_VANTAGE_KEY not set — falling back to TimescaleDB")
        return load_ohlcv_data(ticker)

    try:
        import requests
        base = "https://www.alphavantage.co/query"

        if interval == "daily":
            params = {
                "function":   "TIME_SERIES_DAILY_ADJUSTED",
                "symbol":     ticker,
                "outputsize": "full",
                "apikey":     api_key,
            }
            r    = requests.get(base, params=params, timeout=15)
            data = r.json().get("Time Series (Daily)", {})
            rows = []
            for date_str, vals in data.items():
                rows.append({
                    "Date":   pd.Timestamp(date_str),
                    "Open":   float(vals["1. open"]),
                    "High":   float(vals["2. high"]),
                    "Low":    float(vals["3. low"]),
                    "Close":  float(vals["5. adjusted close"]),
                    "Volume": float(vals["6. volume"]),
                })
        else:
            av_interval = interval.replace("min", "min")
            params = {
                "function":   "TIME_SERIES_INTRADAY",
                "symbol":     ticker,
                "interval":   av_interval,
                "outputsize": "full",
                "apikey":     api_key,
            }
            r    = requests.get(base, params=params, timeout=15)
            key  = f"Time Series ({av_interval})"
            data = r.json().get(key, {})
            rows = []
            for dt_str, vals in data.items():
                rows.append({
                    "Date":   pd.Timestamp(dt_str),
                    "Open":   float(vals["1. open"]),
                    "High":   float(vals["2. high"]),
                    "Low":    float(vals["3. low"]),
                    "Close":  float(vals["4. close"]),
                    "Volume": float(vals["5. volume"]),
                })

        if not rows:
            logger.warning("Alpha Vantage returned no data — falling back to TimescaleDB")
            return load_ohlcv_data(ticker)

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        df["Stock Splits"] = 0
        logger.info(f"Alpha Vantage: fetched {len(df):,} rows for {ticker}")
        return _validate(df)

    except Exception as e:
        logger.warning(f"Alpha Vantage failed ({e}) — falling back to TimescaleDB")
        return load_ohlcv_data(ticker)

# ── Alpaca Markets ────────────────────────────────────────────────────────────
def _load_alpaca(ticker: str, interval: Interval = "daily",
                 days_back: int = 365) -> pd.DataFrame:
    """
    Alpaca Markets REST API.
    """
    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        logger.warning("Alpaca keys not set — falling back to TimescaleDB")
        return load_ohlcv_data(ticker, days_back)

    try:
        import requests
        alpaca_tf_map = {
            "1min": "1Min", "5min": "5Min", "15min": "15Min",
            "30min": "30Min", "60min": "1Hour", "daily": "1Day",
        }
        timeframe = alpaca_tf_map.get(interval, "1Day")
        start     = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end       = datetime.now().strftime("%Y-%m-%d")

        url     = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        headers = {"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret_key}
        params  = {"timeframe": timeframe, "start": start, "end": end,
                   "limit": 10000, "adjustment": "all"}

        rows = []
        while True:
            r    = requests.get(url, headers=headers, params=params, timeout=15)
            body = r.json()
            for bar in body.get("bars", []):
                rows.append({
                    "Date":   pd.Timestamp(bar["t"]).tz_localize(None),
                    "Open":   bar["o"], "High": bar["h"],
                    "Low":    bar["l"], "Close": bar["c"],
                    "Volume": bar["v"],
                })
            next_token = body.get("next_page_token")
            if not next_token:
                break
            params["page_token"] = next_token

        if not rows:
            logger.warning("Alpaca returned no data — falling back to TimescaleDB")
            return load_ohlcv_data(ticker, days_back)

        df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        df["Stock Splits"] = 0
        logger.info(f"Alpaca: fetched {len(df):,} {timeframe} bars for {ticker}")
        return _validate(df)

    except Exception as e:
        logger.warning(f"Alpaca failed ({e}) — falling back to TimescaleDB")
        return load_ohlcv_data(ticker, days_back)

# ── Validation ────────────────────────────────────────────────────────────────
def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV sanity checks — logs warnings, does not raise."""
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
