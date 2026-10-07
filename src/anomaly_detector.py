import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest

from feature_engineering import create_features


FEATURES = [
    "amount",
    "hour",
    "day",
    "is_night",
    "amount_log",
    "amount_to_old_balance",
    "sender_balance_change",
    "sender_balance_error",
    "sender_zero_balance",
    "receiver_balance_change",
    "receiver_balance_error",
    "receiver_zero_balance",
    "is_transfer",
    "is_cash_out",
    "sender_transaction_count",
    "receiver_transaction_count",
    "same_account",
]


def train_anomaly_detector(df):

    X = df[FEATURES]

    print("Training Isolation Forest...")

    detector = IsolationForest(
        n_estimators=100,
        contamination=0.001,
        random_state=42,
        n_jobs=-1
    )

    detector.fit(X)

    return detector


def save_detector(detector):

    joblib.dump(
        detector,
        "models/anomaly_detector.pkl"
    )

    print("Anomaly detector saved to models/anomaly_detector.pkl")


if __name__ == "__main__":

    print("Loading dataset...")

    df = pd.read_csv("data/transactions.csv")

    df = create_features(df)

    detector = train_anomaly_detector(df)

    save_detector(detector)