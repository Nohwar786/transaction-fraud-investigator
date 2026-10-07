import pandas as pd


DATA_PATH = "data/transactions.csv"


def load_transactions(path=DATA_PATH):
    """Load the PaySim transaction dataset."""
    df = pd.read_csv(path)

    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud cases: {df['isFraud'].sum():,}")

    return df


if __name__ == "__main__":
    df = load_transactions()

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())