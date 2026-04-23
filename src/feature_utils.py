"""
Shared feature computation used by app, API, and feature_engineering.
Single source of truth — no more duplication across 3 files.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_features_from_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 51 technical features from a raw OHLCV DataFrame.
    Works on any length (min_periods=1 for rolling windows).
    Safe to call on 10-row input windows.

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume

    Returns
    -------
    DataFrame with all feature columns appended, NaNs filled with 0.
    """
    d = df.copy().astype(float)

    # ── Returns ───────────────────────────────────────────────────────────────
    d["Return"]    = d["Close"].pct_change() * 100
    d["LogReturn"] = np.log(d["Close"] / d["Close"].shift(1))
    d["RangePct"]  = (d["High"] - d["Low"]) / d["Close"].shift(1) * 100

    # ── Lags ──────────────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        d[f"Lag{lag}"] = d["Close"].shift(lag)
    for lag in [1, 2, 3, 5]:
        d[f"RetLag{lag}"] = d["Return"].shift(lag)

    # ── Moving averages ───────────────────────────────────────────────────────
    for w in [5, 7, 10, 21, 50, 200]:
        d[f"MA{w}"] = d["Close"].rolling(w, min_periods=1).mean()

    d["EMA9"]      = d["Close"].ewm(span=9,  adjust=False).mean()
    d["EMA21"]     = d["Close"].ewm(span=21, adjust=False).mean()
    d["EMA_Cross"] = d["EMA9"] - d["EMA21"]

    # ── Rolling stats ─────────────────────────────────────────────────────────
    d["RollingMean_5"]  = d["Close"].rolling(5,  min_periods=1).mean()
    d["RollingMean_10"] = d["Close"].rolling(10, min_periods=1).mean()
    d["RollingStd_5"]   = d["Close"].rolling(5,  min_periods=1).std().fillna(0)
    d["RollingStd_10"]  = d["Close"].rolling(10, min_periods=1).std().fillna(0)
    d["Volatility"]     = d["Return"].rolling(20, min_periods=1).std().fillna(0)
    d["Volatility_5"]   = d["Return"].rolling(5,  min_periods=1).std().fillna(0)
    d["VolRatio_5_20"]  = d["Volatility_5"] / d["Volatility"].replace(0, np.nan)

    # ── Price vs MAs ──────────────────────────────────────────────────────────
    for col in ["MA5", "MA10", "MA21", "MA50", "MA200"]:
        d[f"Price_vs_{col}"] = d["Close"] / d[col].replace(0, np.nan) - 1

    # ── Volume ────────────────────────────────────────────────────────────────
    d["Volume_MA10"]    = d["Volume"].rolling(10, min_periods=1).mean()
    d["Volume_MA20"]    = d["Volume"].rolling(20, min_periods=1).mean()
    d["Volume_Ratio"]   = d["Volume"] / d["Volume_MA10"].replace(0, np.nan)
    d["Volume_Ratio20"] = d["Volume"] / d["Volume_MA20"].replace(0, np.nan)

    obv = (np.sign(d["Close"].diff()) * d["Volume"]).fillna(0).cumsum()
    d["OBV"]       = obv
    d["OBV_MA10"]  = d["OBV"].rolling(10, min_periods=1).mean()
    d["OBV_Ratio"] = d["OBV"] / d["OBV_MA10"].replace(0, np.nan)

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in [7, 14]:
        delta = d["Close"].diff()
        gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
        loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
        rs    = gain / loss.replace(0, np.nan)
        d[f"RSI{period}"] = (100 - (100 / (1 + rs))).fillna(50)
    d["RSI"] = d["RSI14"]

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"]        = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]
    d["MACD_Norm"]   = d["MACD"] / d["Close"].replace(0, np.nan)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid         = d["Close"].rolling(20, min_periods=1).mean()
    bb_std         = d["Close"].rolling(20, min_periods=1).std().fillna(0)
    d["BB_Upper"]  = bb_mid + 2 * bb_std
    d["BB_Lower"]  = bb_mid - 2 * bb_std
    denom          = (d["BB_Upper"] - d["BB_Lower"]).replace(0, np.nan)
    d["BB_Width"]  = denom / bb_mid
    d["BB_Pct"]    = (d["Close"] - d["BB_Lower"]) / denom

    # ── ATR ───────────────────────────────────────────────────────────────────
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift(1)).abs()
    lc  = (d["Low"]  - d["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR"]      = tr.rolling(14, min_periods=1).mean()
    d["ATR_Norm"] = d["ATR"] / d["Close"].replace(0, np.nan)

    # ── Stochastic / Williams / CCI ───────────────────────────────────────────
    low14  = d["Low"].rolling(14,  min_periods=1).min()
    high14 = d["High"].rolling(14, min_periods=1).max()
    rng14  = (high14 - low14).replace(0, np.nan)
    d["Stoch_K"]    = 100 * (d["Close"] - low14) / rng14
    d["Stoch_D"]    = d["Stoch_K"].rolling(3, min_periods=1).mean()
    d["Williams_R"] = -100 * (high14 - d["Close"]) / rng14

    tp     = (d["High"] + d["Low"] + d["Close"]) / 3
    tp_ma  = tp.rolling(20, min_periods=1).mean()
    tp_std = tp.rolling(20, min_periods=1).std().replace(0, np.nan)
    d["CCI"] = (tp - tp_ma) / (0.015 * tp_std)

    # ── Range & momentum ──────────────────────────────────────────────────────
    d["Range"]      = d["High"] - d["Low"]
    d["Range_Norm"] = d["Range"] / d["Close"].replace(0, np.nan)
    d["Momentum5"]  = d["Close"] / d["Close"].shift(5).replace(0,  np.nan) - 1
    d["Momentum10"] = d["Close"] / d["Close"].shift(10).replace(0, np.nan) - 1
    d["Momentum20"] = d["Close"] / d["Close"].shift(20).replace(0, np.nan) - 1

    # ── Calendar ──────────────────────────────────────────────────────────────
    now = pd.Timestamp.now()
    if hasattr(d.index, "dayofweek") and not isinstance(d.index, pd.RangeIndex):
        d["DayOfWeek"]     = d.index.dayofweek
        d["Month"]         = d.index.month
        d["Quarter"]       = d.index.quarter
        d["EarningsMonth"] = d.index.month.isin([1, 4, 7, 10]).astype(int)
    else:
        d["DayOfWeek"]     = now.dayofweek
        d["Month"]         = now.month
        d["Quarter"]       = (now.month - 1) // 3 + 1
        d["EarningsMonth"] = int(now.month in [1, 4, 7, 10])

    # ── Regime defaults (filled by regime_detection if available) ─────────────
    if "Regime" not in d.columns:
        d["Regime"]      = 1
        d["Regime_Bear"] = 0.0
        d["Regime_Side"] = 1.0
        d["Regime_Bull"] = 0.0

    return d.ffill().fillna(0)


def build_prediction_row(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    Compute features and return the last row aligned to model.feature_names_.
    Handles any model version gracefully.
    """
    from src.modeling import FEATURES
    d = compute_features_from_ohlcv(df)
    train_feats = getattr(model, "feature_names_", FEATURES)
    for f in train_feats:
        if f not in d.columns:
            d[f] = 0.0
    return d[train_feats].iloc[[-1]]
