import pandas as pd
import joblib

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


def calculate_risk_score(df, fraud_model, anomaly_model):

    data = df.copy()

    X = data[FEATURES]

    # --------------------------------------------------
    # 1. Fraud model probability
    # --------------------------------------------------

    fraud_probability = fraud_model.predict_proba(X)[:, 1]

    # --------------------------------------------------
    # 2. Anomaly score
    # --------------------------------------------------

    anomaly_raw = anomaly_model.decision_function(X)

    # Convert anomaly score into 0-1 risk
    anomaly_risk = 1 - (
        (anomaly_raw - anomaly_raw.min()) /
        (anomaly_raw.max() - anomaly_raw.min() + 1e-9)
    )

    # --------------------------------------------------
    # 3. Rule-based risk factors
    # --------------------------------------------------

    rule_score = (
        data["is_night"] * 0.15
        + (data["amount_to_old_balance"] > 1).astype(int) * 0.25
        + (data["sender_zero_balance"] == 1).astype(int) * 0.10
        + (data["is_transfer"] == 1).astype(int) * 0.10
        + (data["sender_transaction_count"] > 5).astype(int) * 0.10
    )

    rule_score = rule_score.clip(upper=1.0)

    # --------------------------------------------------
    # 4. Combined risk score
    # --------------------------------------------------

    risk_score = (
        fraud_probability * 0.60
        + anomaly_risk * 0.25
        + rule_score * 0.15
    )

    data["fraud_probability"] = fraud_probability
    data["anomaly_risk"] = anomaly_risk
    data["rule_risk"] = rule_score

    data["risk_score"] = (
        risk_score * 100
    ).round(2)

    # --------------------------------------------------
    # 5. Risk level
    # --------------------------------------------------

    def classify_risk(score):

        if score >= 80:
            return "CRITICAL"

        if score >= 60:
            return "HIGH"

        if score >= 30:
            return "MEDIUM"

        return "LOW"

    data["risk_level"] = data["risk_score"].apply(
        classify_risk
    )

    return data


if __name__ == "__main__":

    print("Loading models...")

    fraud_model = joblib.load(
        "models/fraud_model.pkl"
    )

    anomaly_model = joblib.load(
        "models/anomaly_detector.pkl"
    )

    print("Loading transactions...")

    df = pd.read_csv(
        "data/transactions.csv"
    )

    df = create_features(df)

    print("Calculating risk scores...")

    result = calculate_risk_score(
        df,
        fraud_model,
        anomaly_model
    )

    print("\nRisk distribution:")
    print(
        result["risk_level"].value_counts()
    )

    print("\nHighest-risk transactions:")

    print(
        result[
            [
                "step",
                "type",
                "amount",
                "nameOrig",
                "nameDest",
                "fraud_probability",
                "anomaly_risk",
                "risk_score",
                "risk_level",
                "isFraud",
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
    )

    # Save investigation dataset
    result.to_csv(
        "data/risk_scored_transactions.csv",
        index=False
    )

    print(
        "\nRisk-scored dataset saved to "
        "data/risk_scored_transactions.csv"
    )