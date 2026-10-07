import pandas as pd


def generate_reasons(row):

    reasons = []

    # --------------------------------------------------
    # Machine learning risk
    # --------------------------------------------------

    if row["fraud_probability"] >= 0.80:
        reasons.append(
            "Very high fraud probability"
        )

    elif row["fraud_probability"] >= 0.50:
        reasons.append(
            "Elevated fraud probability"
        )

    # --------------------------------------------------
    # Anomaly detection
    # --------------------------------------------------

    if row["anomaly_risk"] >= 0.75:
        reasons.append(
            "Highly unusual transaction behavior"
        )

    elif row["anomaly_risk"] >= 0.50:
        reasons.append(
            "Unusual transaction behavior"
        )

    # --------------------------------------------------
    # Transaction amount
    # --------------------------------------------------

    if row["amount"] >= 1000000:
        reasons.append(
            "Extremely large transaction amount"
        )

    elif row["amount"] >= 500000:
        reasons.append(
            "Large transaction amount"
        )

    # --------------------------------------------------
    # Transaction type
    # --------------------------------------------------

    if row["is_transfer"] == 1:
        reasons.append(
            "Transfer transaction"
        )

    if row["is_cash_out"] == 1:
        reasons.append(
            "Cash-out transaction"
        )

    # --------------------------------------------------
    # Night transaction
    # --------------------------------------------------

    if row["is_night"] == 1:
        reasons.append(
            "Transaction occurred during unusual hours"
        )

    # --------------------------------------------------
    # Sender balance
    # --------------------------------------------------

    if row["sender_zero_balance"] == 1:
        reasons.append(
            "Sender account was emptied"
        )

    if abs(row["sender_balance_error"]) > 100:
        reasons.append(
            "Unexpected sender balance movement"
        )

    # --------------------------------------------------
    # Receiver balance
    # --------------------------------------------------

    if abs(row["receiver_balance_error"]) > 100:
        reasons.append(
            "Unexpected receiver balance movement"
        )

    # --------------------------------------------------
    # Account activity
    # --------------------------------------------------

    if row["sender_transaction_count"] > 5:
        reasons.append(
            "Sender has unusually high transaction activity"
        )

    if row["receiver_transaction_count"] > 20:
        reasons.append(
            "Receiver is connected to many transactions"
        )

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not reasons:
        reasons.append(
            "No major rule-based warning detected"
        )

    return reasons


def explain_transactions(df):

    data = df.copy()

    data["investigation_reasons"] = data.apply(
        generate_reasons,
        axis=1
    )

    data["investigation_reasons"] = (
        data["investigation_reasons"]
        .apply(lambda x: " | ".join(x))
    )

    return data


if __name__ == "__main__":

    print("Loading risk-scored transactions...")

    df = pd.read_csv(
        "data/risk_scored_transactions.csv"
    )

    result = explain_transactions(df)

    print("\nExample investigation results:")

    print(
        result[
            [
                "amount",
                "type",
                "risk_score",
                "risk_level",
                "investigation_reasons",
                "isFraud",
            ]
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    result.to_csv(
        "data/investigation_transactions.csv",
        index=False
    )

    print(
        "\nInvestigation dataset saved to "
        "data/investigation_transactions.csv"
    )