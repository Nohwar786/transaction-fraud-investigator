import pandas as pd
import numpy as np
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


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


def train_model(df):

    X = df[FEATURES]
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # Classification metrics
    # --------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # Display evaluation
    # --------------------------------------------------

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nModel Evaluation:")
    print(
        f"Precision: {precision:.4f}"
    )
    print(
        f"Recall:    {recall:.4f}"
    )
    print(
        f"F1 Score:  {f1:.4f}"
    )
    print(
        f"ROC-AUC:   {roc_auc:.4f}"
    )
    print(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    # --------------------------------------------------
    # Save evaluation metrics
    # --------------------------------------------------

    evaluation = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
        "fraud_test_samples": int(y_test.sum()),
        "normal_test_samples": int(
            len(y_test) - y_test.sum()
        )
    }

    with open(
        "data/model_evaluation.json",
        "w"
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=4
        )

    print(
        "\nEvaluation saved to "
        "data/model_evaluation.json"
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": model.feature_importances_
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    feature_importance.to_csv(
        "data/feature_importance.csv",
        index=False
    )

    print(
        "Feature importance saved to "
        "data/feature_importance.csv"
    )

    print(
        "\nTop 10 Important Features:"
    )

    print(
        feature_importance.head(10).to_string(
            index=False
        )
    )

    return model


def save_model(model):

    joblib.dump(
        model,
        "models/fraud_model.pkl"
    )

    print(
        "\nModel saved to "
        "models/fraud_model.pkl"
    )


if __name__ == "__main__":

    print(
        "Loading dataset..."
    )

    df = pd.read_csv(
        "data/transactions.csv"
    )

    # --------------------------------------------------
    # Recreate engineered features
    # --------------------------------------------------

    import sys

    sys.path.append(
        "src"
    )

    from feature_engineering import (
        create_features
    )

    df = create_features(
        df
    )

    model = train_model(
        df
    )

    save_model(
        model
    )