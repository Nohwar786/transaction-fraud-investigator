from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import math
import joblib


app = FastAPI(
    title="Fraud Investigation Platform API",
    description="REST API for fraud detection and investigation",
    version="1.0.0"
)


ALERTS_PATH = "data/investigation_alerts.csv"
CASES_PATH = "data/cases.csv"

FRAUD_MODEL_PATH = "models/fraud_model.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_detector.pkl"


# --------------------------------------------------
# Load machine learning models
# --------------------------------------------------

fraud_model = joblib.load(
    FRAUD_MODEL_PATH
)

anomaly_model = joblib.load(
    ANOMALY_MODEL_PATH
)


# --------------------------------------------------
# Feature list
# --------------------------------------------------

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


# --------------------------------------------------
# Prediction request model
# --------------------------------------------------

class TransactionRequest(BaseModel):

    step: int
    transaction_type: str
    amount: float

    sender: str
    receiver: str

    oldbalance_org: float
    newbalance_orig: float

    oldbalance_dest: float
    newbalance_dest: float

    sender_transaction_count: int = 1
    receiver_transaction_count: int = 1


# --------------------------------------------------
# Case request model
# --------------------------------------------------

class CaseRequest(BaseModel):

    transaction_id: str
    transaction_type: str
    amount: float
    sender: str
    receiver: str
    risk_score: float
    risk_level: str
    fraud_probability: float
    case_status: str
    case_priority: str
    investigator_notes: str = ""


# --------------------------------------------------
# JSON-safe conversion
# --------------------------------------------------

def make_json_safe(value):

    if value is None:
        return ""

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return ""

    return value


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Fraud Investigation Platform API",
        "status": "running"
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Get fraud alerts
# --------------------------------------------------

@app.get("/alerts")
def get_alerts():

    if not os.path.exists(ALERTS_PATH):

        return {
            "total_alerts": 0,
            "alerts": []
        }

    if os.path.getsize(ALERTS_PATH) == 0:

        return {
            "total_alerts": 0,
            "alerts": []
        }

    df = pd.read_csv(
        ALERTS_PATH
    )

    alerts = df.to_dict(
        orient="records"
    )

    alerts = [
        {
            key: make_json_safe(value)
            for key, value in record.items()
        }
        for record in alerts
    ]

    return {
        "total_alerts": len(alerts),
        "alerts": alerts
    }


# --------------------------------------------------
# Get investigation cases
# --------------------------------------------------

@app.get("/cases")
def get_cases():

    if not os.path.exists(CASES_PATH):

        return {
            "total_cases": 0,
            "cases": []
        }

    if os.path.getsize(CASES_PATH) == 0:

        return {
            "total_cases": 0,
            "cases": []
        }

    df = pd.read_csv(
        CASES_PATH
    )

    cases = df.to_dict(
        orient="records"
    )

    cases = [
        {
            key: make_json_safe(value)
            for key, value in record.items()
        }
        for record in cases
    ]

    return {
        "total_cases": len(cases),
        "cases": cases
    }


# --------------------------------------------------
# Create investigation case
# --------------------------------------------------

@app.post("/cases")
def create_case(
    case: CaseRequest
):

    case_data = pd.DataFrame(
        [{
            "transaction_id":
                case.transaction_id,

            "transaction_type":
                case.transaction_type,

            "amount":
                case.amount,

            "sender":
                case.sender,

            "receiver":
                case.receiver,

            "risk_score":
                case.risk_score,

            "risk_level":
                case.risk_level,

            "fraud_probability":
                case.fraud_probability,

            "case_status":
                case.case_status,

            "case_priority":
                case.case_priority,

            "investigator_notes":
                case.investigator_notes
        }]
    )


    if (
        os.path.exists(CASES_PATH)
        and os.path.getsize(CASES_PATH) > 0
    ):

        case_data.to_csv(
            CASES_PATH,
            mode="a",
            header=False,
            index=False
        )

    else:

        case_data.to_csv(
            CASES_PATH,
            index=False
        )


    return {
        "message": "Case created successfully",
        "transaction_id":
            case.transaction_id,
        "case_status":
            case.case_status,
        "case_priority":
            case.case_priority
    }


# --------------------------------------------------
# Predict transaction risk
# --------------------------------------------------

@app.post("/predict")
def predict_transaction(
    transaction: TransactionRequest
):

    # ----------------------------------------------
    # Basic transaction values
    # ----------------------------------------------

    step = transaction.step

    amount = transaction.amount

    oldbalance_org = (
        transaction.oldbalance_org
    )

    newbalance_orig = (
        transaction.newbalance_orig
    )

    oldbalance_dest = (
        transaction.oldbalance_dest
    )

    newbalance_dest = (
        transaction.newbalance_dest
    )


    # ----------------------------------------------
    # Time features
    # ----------------------------------------------

    hour = step % 24

    day = step // 24

    is_night = int(
        hour in [0, 1, 2, 3, 4, 5]
    )


    # ----------------------------------------------
    # Amount features
    # ----------------------------------------------

    amount_log = np.log1p(
        amount
    )

    amount_to_old_balance = (
        amount /
        (oldbalance_org + 1)
    )


    # ----------------------------------------------
    # Sender balance features
    # ----------------------------------------------

    sender_balance_change = (
        oldbalance_org -
        newbalance_orig
    )

    sender_balance_error = (
        oldbalance_org -
        amount -
        newbalance_orig
    )

    sender_zero_balance = int(
        newbalance_orig == 0
    )


    # ----------------------------------------------
    # Receiver balance features
    # ----------------------------------------------

    receiver_balance_change = (
        newbalance_dest -
        oldbalance_dest
    )

    receiver_balance_error = (
        oldbalance_dest +
        amount -
        newbalance_dest
    )

    receiver_zero_balance = int(
        newbalance_dest == 0
    )


    # ----------------------------------------------
    # Transaction type features
    # ----------------------------------------------

    is_transfer = int(
        transaction.transaction_type
        == "TRANSFER"
    )

    is_cash_out = int(
        transaction.transaction_type
        == "CASH_OUT"
    )


    # ----------------------------------------------
    # Account features
    # ----------------------------------------------

    sender_transaction_count = (
        transaction.sender_transaction_count
    )

    receiver_transaction_count = (
        transaction.receiver_transaction_count
    )

    same_account = int(
        transaction.sender ==
        transaction.receiver
    )


    # ----------------------------------------------
    # Create model input
    # ----------------------------------------------

    model_input = pd.DataFrame(
        [[
            amount,
            hour,
            day,
            is_night,
            amount_log,
            amount_to_old_balance,
            sender_balance_change,
            sender_balance_error,
            sender_zero_balance,
            receiver_balance_change,
            receiver_balance_error,
            receiver_zero_balance,
            is_transfer,
            is_cash_out,
            sender_transaction_count,
            receiver_transaction_count,
            same_account
        ]],
        columns=FEATURES
    )


    # ----------------------------------------------
    # Fraud model prediction
    # ----------------------------------------------

    fraud_probability = float(
        fraud_model.predict_proba(
            model_input
        )[0][1]
    )


    # ----------------------------------------------
    # Anomaly model prediction
    # ----------------------------------------------

    anomaly_prediction = (
        anomaly_model.predict(
            model_input
        )[0]
    )

    anomaly_score = (
        anomaly_model.decision_function(
            model_input
        )[0]
    )


    # Convert anomaly score into
    # a simple 0-1 anomaly risk
    anomaly_risk = float(
        np.clip(
            0.5 - anomaly_score,
            0,
            1
        )
    )


    # ----------------------------------------------
    # Rule-based risk
    # ----------------------------------------------

    rule_score = (

        is_night * 0.15

        + int(
            amount_to_old_balance > 1
        ) * 0.25

        + sender_zero_balance * 0.10

        + is_transfer * 0.10

        + int(
            sender_transaction_count > 5
        ) * 0.10
    )


    rule_score = min(
        rule_score,
        1.0
    )


    # ----------------------------------------------
    # Combined risk score
    # ----------------------------------------------

    risk_score = (

        fraud_probability * 0.60

        + anomaly_risk * 0.25

        + rule_score * 0.15
    ) * 100


    risk_score = round(
        float(risk_score),
        2
    )


    # ----------------------------------------------
    # Risk level
    # ----------------------------------------------

    if risk_score >= 80:

        risk_level = "CRITICAL"

    elif risk_score >= 60:

        risk_level = "HIGH"

    elif risk_score >= 30:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ----------------------------------------------
    # Response
    # ----------------------------------------------

    return {
        "transaction": {
            "type":
                transaction.transaction_type,

            "amount":
                amount,

            "sender":
                transaction.sender,

            "receiver":
                transaction.receiver
        },

        "fraud_probability":
            round(
                fraud_probability,
                6
            ),

        "anomaly_risk":
            round(
                anomaly_risk,
                6
            ),

        "rule_risk":
            round(
                rule_score,
                6
            ),

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "anomaly_prediction":
            int(anomaly_prediction)
    }