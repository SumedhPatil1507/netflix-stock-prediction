import numpy as np
import pandas as pd


def create_features(df):
    # ── Returns ──────────────────────────────────────────────────────────────
    df['Return']    = df['Close'].pct_change() * 100
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RangePct']  = (df['High'] - df['Low']) / df['Close'].shift(1) * 100

    # ── Lag features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f'Lag{lag}'] = df['Close'].shift(lag)

    # ── Moving averages ───────────────────────────────────────────────────────
    for w in [7, 21, 50, 200]:
        df[f'MA{w}'] = df['Close'].rolling(w).mean()

    # ── Rolling stats ─────────────────────────────────────────────────────────
    df['RollingMean_5']  = df['Close'].rolling(5).mean()
    df['RollingMean_10'] = df['Close'].rolling(10).mean()
    df['RollingStd_5']   = df['Close'].rolling(5).std()
    df['RollingStd_10']  = df['Close'].rolling(10).std()
    df['Volatility']     = df['Return'].rolling(20).std()

    # ── Volume features ───────────────────────────────────────────────────────
    df['Volume_MA10']    = df['Volume'].rolling(10).mean()
    df['Volume_Ratio']   = df['Volume'] / df['Volume_MA10']

    # ── RSI (14-period) ───────────────────────────────────────────────────────
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12        = df['Close'].ewm(span=12, adjust=False).mean()
    ema26        = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid          = df['Close'].rolling(20).mean()
    bb_std          = df['Close'].rolling(20).std()
    df['BB_Upper']  = bb_mid + 2 * bb_std
    df['BB_Lower']  = bb_mid - 2 * bb_std
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / bb_mid
    df['BB_Pct']    = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # ── ATR (14-period) ───────────────────────────────────────────────────────
    high_low   = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close  = (df['Low']  - df['Close'].shift(1)).abs()
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR']  = tr.rolling(14).mean()

    # ── Price position ────────────────────────────────────────────────────────
    df['Range']         = df['High'] - df['Low']
    df['Price_vs_MA50'] = df['Close'] / df['MA50'] - 1
    df['Price_vs_MA200']= df['Close'] / df['MA200'] - 1

    # ── Calendar features ─────────────────────────────────────────────────────
    df['DayOfWeek'] = df.index.dayofweek
    df['Month']     = df.index.month

    return df.dropna()
