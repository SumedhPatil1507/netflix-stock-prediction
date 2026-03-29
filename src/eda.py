def basic_eda(df):
    print("\n📊 Dataset Info")
    print(df.info())

    print("\n📈 Statistical Summary")
    print(df.describe())

    print("\n❗ Missing Values")
    print(df.isnull().sum())