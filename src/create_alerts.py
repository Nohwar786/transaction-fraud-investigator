import pandas as pd

INPUT_PATH = "data/investigation_transactions.csv"
OUTPUT_PATH = "data/investigation_alerts.csv"


print("Loading investigation data...")

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} transactions")


alerts = df[
    df["risk_level"].isin(["HIGH", "CRITICAL"])
].copy()


alerts = alerts.sort_values(
    "risk_score",
    ascending=False
)


alerts.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nAlert dataset created successfully.")

print(f"Total alerts: {len(alerts):,}")

print("\nRisk distribution:")
print(
    alerts["risk_level"].value_counts()
)

print(
    f"\nSaved to: {OUTPUT_PATH}"
)