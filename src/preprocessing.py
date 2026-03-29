import pandas as pd

def preprocess_data(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date').sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    return df