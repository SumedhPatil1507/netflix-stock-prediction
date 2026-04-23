from __future__ import annotations
import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Core returns ──────────────────────────────────────────────────────────
    df['Return']     = df['Close'].pct_change() * 100
    df['LogReturn']  = np.log(df['Close'] / df['Close'].shift(1))
    df['RangePct']   = (df['High'] - df['Low']) / df['Close'].shift(1) * 100

    # ── Lag close prices ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'Lag{lag}'] = df['Close'].shift(lag)

    # ── Lag returns (more stationary than price lags) ─────────────────────────
    for lag in [1, 2, 3, 5]:
        df[f'RetLag{lag}'] = df['Return'].shift(lag)

    # ── Moving averages ───────────────────────────────────────────────────────
    for w in [5, 7, 10, 21, 50, 200]:
        df[f'MA{w}'] = df['Close'].rolling(w).mean()

    # ── EMA crossover signals ─────────────────────────────────────────────────
    df['EMA9']  = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_Cross'] = df['EMA9'] - df['EMA21']   # positive = bullish

    # ── Rolling stats ─────────────────────────────────────────────────────────
    df['RollingMean_5']  = df['Close'].rolling(5).mean()
    df['RollingMean_10'] = df['Close'].rolling(10).mean()
    df['RollingStd_5']   = df['Close'].rolling(5).std()
    df['RollingStd_10']  = df['Close'].rolling(10).std()
    df['Volatility']     = df['Return'].rolling(20).std()
    df['Volatility_5']   = df['Return'].rolling(5).std()

    # ── Price position relative to MAs ───────────────────────────────────────
    df['Price_vs_MA5']   = df['Close'] / df['MA5']   - 1
    df['Price_vs_MA10']  = df['Close'] / df['MA10']  - 1
    df['Price_vs_MA21']  = df['Close'] / df['MA21']  - 1
    df['Price_vs_MA50']  = df['Close'] / df['MA50']  - 1
    df['Price_vs_MA200'] = df['Close'] / df['MA200'] - 1

    # ── Volume features ───────────────────────────────────────────────────────
    df['Volume_MA10']   = df['Volume'].rolling(10).mean()
    df['Volume_MA20']   = df['Volume'].rolling(20).mean()
    df['Volume_Ratio']  = df['Volume'] / df['Volume_MA10']
    df['Volume_Ratio20']= df['Volume'] / df['Volume_MA20']

    # On-Balance Volume (OBV)
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = obv
    df['OBV_MA10'] = df['OBV'].rolling(10).mean()
    df['OBV_Ratio'] = df['OBV'] / df['OBV_MA10'].replace(0, np.nan)

    # ── RSI (14 & 7) ──────────────────────────────────────────────────────────
    for period in [7, 14]:
        delta = df['Close'].diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f'RSI{period}'] = 100 - (100 / (1 + rs))

    df['RSI'] = df['RSI14']   # alias for backward compat

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']
    df['MACD_Norm']   = df['MACD'] / df['Close']   # price-normalised

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid         = df['Close'].rolling(20).mean()
    bb_std         = df['Close'].rolling(20).std()
    df['BB_Upper'] = bb_mid + 2 * bb_std
    df['BB_Lower'] = bb_mid - 2 * bb_std
    denom          = (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)
    df['BB_Width'] = denom / bb_mid
    df['BB_Pct']   = (df['Close'] - df['BB_Lower']) / denom

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR']      = tr.rolling(14).mean()
    df['ATR_Norm'] = df['ATR'] / df['Close']   # normalised

    # ── Stochastic Oscillator %K / %D ────────────────────────────────────────
    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14).replace(0, np.nan)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────────
    df['Williams_R'] = -100 * (high14 - df['Close']) / (high14 - low14).replace(0, np.nan)

    # ── CCI (Commodity Channel Index) ─────────────────────────────────────────
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_std = tp.rolling(20).std()
    df['CCI'] = (tp - tp_ma) / (0.015 * tp_std.replace(0, np.nan))

    # ── Price range & momentum ────────────────────────────────────────────────
    df['Range']       = df['High'] - df['Low']
    df['Range_Norm']  = df['Range'] / df['Close']
    df['Momentum5']   = df['Close'] / df['Close'].shift(5)  - 1
    df['Momentum10']  = df['Close'] / df['Close'].shift(10) - 1
    df['Momentum20']  = df['Close'] / df['Close'].shift(20) - 1

    # ── Calendar ──────────────────────────────────────────────────────────────
    df['DayOfWeek'] = df.index.dayofweek
    df['Month']     = df.index.month
    df['Quarter']   = df.index.quarter

    # ── Earnings proximity flag (approx: Jan/Apr/Jul/Oct earnings months) ─────
    df['EarningsMonth'] = df['Month'].isin([1, 4, 7, 10]).astype(int)

    # ── Realised volatility ratio (short vs long) — regime proxy ─────────────
    df['VolRatio_5_20'] = df['Volatility_5'] / df['Volatility'].replace(0, np.nan)

    return df.dropna()
