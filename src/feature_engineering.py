import numpy as np

def create_features(df):
    df['Return'] = df['Close'].pct_change() * 100
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df['RangePct'] = (df['High'] - df['Low']) / df['Close'].shift(1) * 100

    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    df['Volatility'] = df['Return'].rolling(20).std()

    return df.dropna()