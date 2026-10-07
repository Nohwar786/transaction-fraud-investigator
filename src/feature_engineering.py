import pandas as pd
import numpy as np


def create_features(df):
    """Create investigation-focused transaction features."""

    data = df.copy()

    # --------------------------------------------------
    # 1. Time features
    # PaySim 'step' represents one hour
    # --------------------------------------------------
    data["hour"] = data["step"] % 24
    data["day"] = data["step"] // 24

    # Night-time transaction indicator
    data["is_night"] = data["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    # --------------------------------------------------
    # 2. Transaction amount features
    # --------------------------------------------------
    data["amount_log"] = np.log1p(data["amount"])

    data["amount_to_old_balance"] = (
        data["amount"] /
        (data["oldbalanceOrg"] + 1)
    )

    # --------------------------------------------------
    # 3. Sender balance features
    # --------------------------------------------------
    data["sender_balance_change"] = (
        data["oldbalanceOrg"] -
        data["newbalanceOrig"]
    )

    data["sender_balance_error"] = (
        data["oldbalanceOrg"] -
        data["amount"] -
        data["newbalanceOrig"]
    )

    data["sender_zero_balance"] = (
        data["newbalanceOrig"] == 0
    ).astype(int)

    # --------------------------------------------------
    # 4. Receiver balance features
    # --------------------------------------------------
    data["receiver_balance_change"] = (
        data["newbalanceDest"] -
        data["oldbalanceDest"]
    )

    data["receiver_balance_error"] = (
        data["oldbalanceDest"] +
        data["amount"] -
        data["newbalanceDest"]
    )

    data["receiver_zero_balance"] = (
        data["newbalanceDest"] == 0
    ).astype(int)

    # --------------------------------------------------
    # 5. Transaction type indicators
    # --------------------------------------------------
    data["is_transfer"] = (
        data["type"] == "TRANSFER"
    ).astype(int)

    data["is_cash_out"] = (
        data["type"] == "CASH_OUT"
    ).astype(int)

    # --------------------------------------------------
    # 6. Account activity
    # --------------------------------------------------
    sender_counts = data["nameOrig"].value_counts()
    receiver_counts = data["nameDest"].value_counts()

    data["sender_transaction_count"] = (
        data["nameOrig"].map(sender_counts)
    )

    data["receiver_transaction_count"] = (
        data["nameDest"].map(receiver_counts)
    )

    # --------------------------------------------------
    # 7. Account relationship
    # --------------------------------------------------
    data["same_account"] = (
        data["nameOrig"] == data["nameDest"]
    ).astype(int)

    print("Feature engineering completed.")
    print(f"Original features: {len(df.columns)}")
    print(f"New features: {len(data.columns)}")

    return data


if __name__ == "__main__":
    df = pd.read_csv("data/transactions.csv")

    featured_df = create_features(df)

    print("\nNew feature columns:")
    print(featured_df.columns.tolist())

    print("\nSample engineered features:")
    print(
        featured_df[
            [
                "amount",
                "hour",
                "is_night",
                "amount_to_old_balance",
                "sender_balance_change",
                "receiver_balance_change",
                "sender_transaction_count",
                "receiver_transaction_count",
            ]
        ].head()
    )