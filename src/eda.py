import pandas as pd


def basic_eda(df: pd.DataFrame):
    print("\n📊 Dataset Info")
    print(df.info())

    print("\n📈 Statistical Summary")
    print(df.describe())

    print("\n❗ Missing Values")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "  None")

    print(f"\n📅 Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"   Rows: {len(df):,}")
