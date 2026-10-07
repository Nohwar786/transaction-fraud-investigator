import os
import sys
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# --------------------------------------------------
# API client
# --------------------------------------------------

from api_client import (
    get_alerts,
    get_cases,
    predict_transaction,
    create_case
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Investigation Platform",
    page_icon="🚨",
    layout="wide"
)


# --------------------------------------------------
# Constants
# --------------------------------------------------

TOTAL_TRANSACTIONS = 6362620
CONFIRMED_FRAUD = 8213

NETWORK_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "network_investigation.csv"
)


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "prediction_transaction" not in st.session_state:
    st.session_state["prediction_transaction"] = None

if "prediction_case_created" not in st.session_state:
    st.session_state["prediction_case_created"] = False

if "prediction_case_message" not in st.session_state:
    st.session_state["prediction_case_message"] = ""


# --------------------------------------------------
# Load alerts from FastAPI
# --------------------------------------------------

@st.cache_data(ttl=30)
def load_alerts():

    response = get_alerts()

    return pd.DataFrame(
        response["alerts"]
    )


# --------------------------------------------------
# Load cases from FastAPI
# --------------------------------------------------

@st.cache_data(ttl=10)
def load_cases():

    response = get_cases()

    return pd.DataFrame(
        response["cases"]
    )


# --------------------------------------------------
# Load network data
# --------------------------------------------------

@st.cache_data
def load_network_data():

    if not os.path.exists(NETWORK_PATH):
        return pd.DataFrame()

    return pd.read_csv(
        NETWORK_PATH
    )


# --------------------------------------------------
# Load application data
# --------------------------------------------------

try:

    df = load_alerts()

except Exception as e:

    st.error(
        f"Unable to connect to FastAPI: {e}"
    )

    st.info(
        "Make sure FastAPI is running with: "
        "uvicorn api.main:app --reload"
    )

    st.stop()


network_df = load_network_data()


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title(
    "🚨 Fraud Investigation Platform"
)

st.write(
    "Multi-layer fraud detection and transaction "
    "investigation system"
)


# --------------------------------------------------
# Top metrics
# --------------------------------------------------

critical_transactions = int(
    (
        df["risk_level"] == "CRITICAL"
    ).sum()
)

high_risk_transactions = int(
    (
        df["risk_level"] == "HIGH"
    ).sum()
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Total Transactions",
        f"{TOTAL_TRANSACTIONS:,}"
    )


with metric2:

    st.metric(
        "Confirmed Fraud",
        f"{CONFIRMED_FRAUD:,}"
    )


with metric3:

    st.metric(
        "Critical Risk",
        f"{critical_transactions:,}"
    )


with metric4:

    st.metric(
        "High Risk",
        f"{high_risk_transactions:,}"
    )


st.divider()


# ==================================================
# Investigation Summary
# ==================================================

st.subheader(
    "📊 Investigation Summary"
)


critical_alerts = df[
    df["risk_level"] == "CRITICAL"
]


high_alerts = df[
    df["risk_level"] == "HIGH"
]


highest_risk_score = df[
    "risk_score"
].max()


highest_transaction_amount = df[
    "amount"
].max()


summary1, summary2, summary3, summary4 = st.columns(4)


with summary1:

    st.metric(
        "Critical Alerts",
        f"{len(critical_alerts):,}"
    )


with summary2:

    st.metric(
        "High-Risk Alerts",
        f"{len(high_alerts):,}"
    )


with summary3:

    st.metric(
        "Highest Risk Score",
        f"{highest_risk_score:.2f}/100"
    )


with summary4:

    st.metric(
        "Highest Transaction",
        f"₹{highest_transaction_amount:,.0f}"
    )


st.divider()


# ==================================================
# Risk Distribution
# ==================================================

st.subheader(
    "Risk Distribution"
)


risk_distribution = (
    df["risk_level"]
    .value_counts()
    .reindex(
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        fill_value=0
    )
)


st.bar_chart(
    risk_distribution
)


st.divider()


# ==================================================
# Suspicious Activity Over Time
# ==================================================

st.subheader(
    "📈 Suspicious Activity Over Time"
)


time_risk = (
    df.groupby("step")
    .agg(
        alerts=("risk_score", "count"),
        average_risk=("risk_score", "mean")
    )
    .reset_index()
)


if len(time_risk) > 0:

    st.line_chart(
        time_risk.set_index(
            "step"
        )["alerts"]
    )

else:

    st.info(
        "No time-series data available."
    )


st.divider()


# ==================================================
# Top Suspicious Accounts
# ==================================================

st.subheader(
    "👤 Top Suspicious Accounts"
)


sender_counts = (
    df["nameOrig"]
    .value_counts()
)


receiver_counts = (
    df["nameDest"]
    .value_counts()
)


account_counts = (
    pd.concat(
        [
            sender_counts,
            receiver_counts
        ]
    )
    .groupby(level=0)
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


account_table = (
    account_counts
    .rename(
        "Suspicious Transactions"
    )
    .reset_index()
)


account_table.columns = [
    "Account",
    "Suspicious Transactions"
]


if not network_df.empty:

    sender_network_counts = (
        network_df
        .groupby("nameOrig")[
            "sender_network_degree"
        ]
        .max()
    )

    receiver_network_counts = (
        network_df
        .groupby("nameDest")[
            "receiver_network_degree"
        ]
        .max()
    )

    account_table[
        "Network Degree"
    ] = (
        account_table["Account"]
        .map(
            sender_network_counts
        )
        .fillna(
            account_table["Account"].map(
                receiver_network_counts
            )
        )
        .fillna(0)
        .astype(int)
    )

else:

    account_table[
        "Network Degree"
    ] = 0


st.dataframe(
    account_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==================================================
# Transaction Investigation
# ==================================================

st.subheader(
    "🔎 Transaction Investigation"
)


filter_col1, filter_col2 = st.columns(2)


with filter_col1:

    risk_filter = st.multiselect(
        "Risk Level",
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW"
        ],
        default=[
            "CRITICAL",
            "HIGH"
        ]
    )


with filter_col2:

    transaction_type_filter = st.multiselect(
        "Transaction Type",
        sorted(
            df["type"]
            .dropna()
            .unique()
            .tolist()
        ),
        default=sorted(
            df["type"]
            .dropna()
            .unique()
            .tolist()
        )
    )


search_account = st.text_input(
    "Search sender or receiver account",
    placeholder="Example: C1191696703"
)


filtered_df = df[
    df["risk_level"].isin(
        risk_filter
    )
    &
    df["type"].isin(
        transaction_type_filter
    )
].copy()


if search_account:

    search_account = (
        search_account
        .strip()
    )

    filtered_df = filtered_df[
        filtered_df["nameOrig"]
        .astype(str)
        .str.contains(
            search_account,
            case=False,
            na=False
        )
        |
        filtered_df["nameDest"]
        .astype(str)
        .str.contains(
            search_account,
            case=False,
            na=False
        )
    ]


st.write(
    f"Investigation alerts found: "
    f"**{len(filtered_df):,}**"
)


if len(filtered_df) > 0:

    display_columns = [
        "nameOrig",
        "nameDest",
        "type",
        "amount",
        "risk_score",
        "risk_level",
        "fraud_probability"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[
            available_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(100),
        use_container_width=True,
        hide_index=True
    )

    transaction_index = st.selectbox(
        "Select Transaction",
        filtered_df.index.tolist(),
        format_func=lambda index:
            (
                f"{filtered_df.loc[index, 'nameOrig']} → "
                f"{filtered_df.loc[index, 'nameDest']} | "
                f"{filtered_df.loc[index, 'type']} | "
                f"₹{filtered_df.loc[index, 'amount']:,.2f} | "
                f"Risk {filtered_df.loc[index, 'risk_score']:.2f}"
            ),
        key="selected_transaction"
    )

    transaction = filtered_df.loc[
        transaction_index
    ]


    # --------------------------------------------------
    # Selected transaction details
    # --------------------------------------------------

    st.write(
        "### 🚨 Selected Transaction"
    )


    detail1, detail2, detail3, detail4 = st.columns(4)


    with detail1:

        st.metric(
            "Amount",
            f"₹{transaction['amount']:,.2f}"
        )


    with detail2:

        st.metric(
            "Risk Score",
            f"{transaction['risk_score']:.2f}/100"
        )


    with detail3:

        st.metric(
            "Risk Level",
            transaction["risk_level"]
        )


    with detail4:

        st.metric(
            "Fraud Probability",
            f"{transaction['fraud_probability'] * 100:.2f}%"
        )


    st.write(
        f"**Sender:** {transaction['nameOrig']}"
    )

    st.write(
        f"**Receiver:** {transaction['nameDest']}"
    )

    st.write(
        f"**Transaction Type:** {transaction['type']}"
    )


    st.divider()


    # ==================================================
    # Network Investigation
    # ==================================================

    st.write(
        "### 🌐 Network Investigation"
    )


    sender = transaction["nameOrig"]

    receiver = transaction["nameDest"]


    if not network_df.empty:

        sender_network = network_df[
            network_df["nameOrig"]
            == sender
        ]

        receiver_network = network_df[
            network_df["nameDest"]
            == receiver
        ]


        network_col1, network_col2 = st.columns(2)


        with network_col1:

            st.write(
                "**Sender Network**"
            )

            if len(sender_network) > 0:

                sender_degree = int(
                    sender_network[
                        "sender_network_degree"
                    ].max()
                )

                st.metric(
                    "Sender Network Degree",
                    sender_degree
                )

            else:

                st.info(
                    "No suspicious network "
                    "connection found."
                )


        with network_col2:

            st.write(
                "**Receiver Network**"
            )

            if len(receiver_network) > 0:

                receiver_degree = int(
                    receiver_network[
                        "receiver_network_degree"
                    ].max()
                )

                st.metric(
                    "Receiver Network Degree",
                    receiver_degree
                )

            else:

                st.info(
                    "No suspicious network "
                    "connection found."
                )


        # --------------------------------------------------
        # Suspicious network graph
        # --------------------------------------------------

        st.write(
            "### 🔗 Suspicious Network Graph"
        )


        graph_data = network_df[
            (network_df["nameOrig"] == sender)
            |
            (network_df["nameDest"] == sender)
            |
            (network_df["nameOrig"] == receiver)
            |
            (network_df["nameDest"] == receiver)
        ].copy()


        graph_data = (
            graph_data
            .sort_values(
                "risk_score",
                ascending=False
            )
            .head(20)
        )


        if len(graph_data) > 0:

            G = nx.DiGraph()


            for _, row in graph_data.iterrows():

                G.add_edge(
                    row["nameOrig"],
                    row["nameDest"],
                    risk=row["risk_score"],
                    amount=row["amount"]
                )


            if len(G.nodes) > 0:

                fig, ax = plt.subplots(
                    figsize=(10, 7)
                )


                pos = nx.spring_layout(
                    G,
                    seed=42,
                    k=1.5
                )


                nx.draw_networkx(
                    G,
                    pos,
                    ax=ax,
                    with_labels=True,
                    node_size=1800,
                    font_size=8,
                    arrows=True,
                    width=1.5
                )


                ax.set_title(
                    "Suspicious Transaction Network"
                )

                ax.axis("off")


                st.pyplot(
                    fig,
                    use_container_width=True
                )


                plt.close(fig)


                st.caption(
                    "Network shows up to 20 highest-risk "
                    "suspicious connections around the "
                    "selected transaction."
                )

            else:

                st.info(
                    "No network graph available."
                )

        else:

            st.info(
                "No connected suspicious transactions "
                "found for this account."
            )

    else:

        st.info(
            "Network investigation data is unavailable."
        )


    st.divider()


    # ==================================================
    # Investigation Reasons
    # ==================================================

    st.write(
        "### 🧠 Investigation Reasons"
    )


    reasons = []


    fraud_probability = float(
        transaction["fraud_probability"]
    )


    risk_score = float(
        transaction["risk_score"]
    )


    if fraud_probability >= 0.90:

        reasons.append(
            "Very high fraud probability."
        )

    elif fraud_probability >= 0.70:

        reasons.append(
            "Elevated fraud probability."
        )


    if "anomaly_risk" in transaction:

        anomaly_risk = float(
            transaction["anomaly_risk"]
        )

        if anomaly_risk >= 0.70:

            reasons.append(
                "High anomaly risk detected."
            )

        elif anomaly_risk >= 0.40:

            reasons.append(
                "Elevated anomaly risk detected."
            )


    if transaction["amount"] >= 1000000:

        reasons.append(
            "Very large transaction amount."
        )

    elif transaction["amount"] >= 100000:

        reasons.append(
            "Large transaction amount."
        )


    if transaction["type"] == "TRANSFER":

        reasons.append(
            "Transfer transaction requires review."
        )


    if transaction["type"] == "CASH_OUT":

        reasons.append(
            "Cash-out transaction requires review."
        )


    if "is_night" in transaction:

        if int(transaction["is_night"]) == 1:

            reasons.append(
                "Transaction occurred during unusual night hours."
            )


    if "sender_zero_balance" in transaction:

        if int(transaction["sender_zero_balance"]) == 1:

            reasons.append(
                "Sender balance reached zero."
            )


    if "sender_balance_error" in transaction:

        if abs(
            float(transaction["sender_balance_error"])
        ) > 1:

            reasons.append(
                "Sender balance inconsistency detected."
            )


    if "receiver_balance_error" in transaction:

        if abs(
            float(transaction["receiver_balance_error"])
        ) > 1:

            reasons.append(
                "Receiver balance inconsistency detected."
            )


    if "sender_transaction_count" in transaction:

        if int(
            transaction["sender_transaction_count"]
        ) > 5:

            reasons.append(
                "Sender has unusually high transaction activity."
            )


    if "receiver_transaction_count" in transaction:

        if int(
            transaction["receiver_transaction_count"]
        ) > 5:

            reasons.append(
                "Receiver has unusually high transaction activity."
            )


    if len(reasons) == 0:

        reasons.append(
            "No major rule-based warning identified."
        )


    for reason in reasons:

        st.write(
            f"• {reason}"
        )


    st.divider()


    # ==================================================
    # Model Assessment
    # ==================================================

    st.write(
        "### 🤖 Model Assessment"
    )


    model_col1, model_col2, model_col3 = st.columns(3)


    with model_col1:

        st.metric(
            "Fraud Model",
            f"{fraud_probability * 100:.2f}%"
        )


    with model_col2:

        if "anomaly_risk" in transaction:

            st.metric(
                "Anomaly Model",
                f"{float(transaction['anomaly_risk']) * 100:.2f}%"
            )

        else:

            st.metric(
                "Anomaly Model",
                "N/A"
            )


    with model_col3:

        st.metric(
            "Combined Risk",
            f"{risk_score:.2f}/100"
        )


    st.divider()


    # ==================================================
    # Case Management
    # ==================================================

    st.write(
        "### 🗂️ Case Management"
    )


    case_col1, case_col2 = st.columns(2)


    with case_col1:

        case_status = st.selectbox(
            "Case Status",
            [
                "Flagged",
                "Under Investigation",
                "Cleared"
            ],
            key=f"case_status_{transaction_index}"
        )


    with case_col2:

        if transaction["risk_level"] == "CRITICAL":

            default_priority = 0

        elif transaction["risk_level"] == "HIGH":

            default_priority = 1

        elif transaction["risk_level"] == "MEDIUM":

            default_priority = 2

        else:

            default_priority = 3


        case_priority = st.selectbox(
            "Case Priority",
            [
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            index=default_priority,
            key=f"case_priority_{transaction_index}"
        )


    case_notes = st.text_area(
        "Investigator Notes",
        placeholder="Enter investigation notes...",
        key=f"case_notes_{transaction_index}"
    )


    if st.button(
        "💾 Save Case",
        key=f"save_case_{transaction_index}"
    ):

        case_data = {

            "transaction_id":
                transaction["nameOrig"]
                + "_"
                + str(transaction["step"]),

            "transaction_type":
                transaction["type"],

            "amount":
                float(transaction["amount"]),

            "sender":
                transaction["nameOrig"],

            "receiver":
                transaction["nameDest"],

            "risk_score":
                float(transaction["risk_score"]),

            "risk_level":
                transaction["risk_level"],

            "fraud_probability":
                float(
                    transaction["fraud_probability"]
                ),

            "case_status":
                case_status,

            "case_priority":
                case_priority,

            "investigator_notes":
                case_notes
        }


        try:

            response = create_case(
                case_data
            )

            load_cases.clear()

            st.success(
                "Case saved successfully: "
                f"{response['transaction_id']} | "
                f"{response['case_status']} | "
                f"Priority: "
                f"{response['case_priority']}"
            )

        except Exception as e:

            st.error(
                f"Failed to save case: {e}"
            )


else:

    st.info(
        "No transactions match the selected filters."
    )


st.divider()


# ==================================================
# Case History
# ==================================================

st.subheader(
    "📁 Case History"
)


try:

    cases_response = get_cases()

    cases_df = pd.DataFrame(
        cases_response["cases"]
    )


    if len(cases_df) > 0:

        history_col1, history_col2, history_col3, history_col4 = (
            st.columns(4)
        )


        with history_col1:

            st.metric(
                "Total Cases",
                len(cases_df)
            )


        with history_col2:

            st.metric(
                "Flagged",
                int(
                    (
                        cases_df["case_status"]
                        == "Flagged"
                    ).sum()
                )
            )


        with history_col3:

            st.metric(
                "Under Investigation",
                int(
                    (
                        cases_df["case_status"]
                        == "Under Investigation"
                    ).sum()
                )
            )


        with history_col4:

            st.metric(
                "Cleared",
                int(
                    (
                        cases_df["case_status"]
                        == "Cleared"
                    ).sum()
                )
            )


        case_status_filter = st.multiselect(
            "Filter Case Status",
            [
                "Flagged",
                "Under Investigation",
                "Cleared"
            ],
            default=[
                "Flagged",
                "Under Investigation",
                "Cleared"
            ],
            key="case_history_status"
        )


        filtered_cases = cases_df[
            cases_df["case_status"].isin(
                case_status_filter
            )
        ]


        history_columns = [
            "transaction_id",
            "transaction_type",
            "amount",
            "risk_score",
            "risk_level",
            "case_status",
            "case_priority",
            "investigator_notes"
        ]


        available_history_columns = [
            column
            for column in history_columns
            if column in filtered_cases.columns
        ]


        st.dataframe(
            filtered_cases[
                available_history_columns
            ],
            use_container_width=True,
            hide_index=True
        )


    else:

        st.info(
            "No investigation cases have been saved yet."
        )


except Exception as e:

    st.error(
        f"Failed to load case history: {e}"
    )


st.divider()


# ==================================================
# New Transaction Prediction
# ==================================================

st.subheader(
    "🤖 Predict New Transaction"
)


st.write(
    "Enter transaction details below to generate "
    "a real-time fraud risk assessment using "
    "the FastAPI prediction service."
)


# --------------------------------------------------
# Start New Prediction
# --------------------------------------------------

if st.button(
    "🆕 Start New Prediction",
    key="start_new_prediction"
):

    st.session_state[
        "prediction_result"
    ] = None

    st.session_state[
        "prediction_transaction"
    ] = None

    st.session_state[
        "prediction_case_created"
    ] = False

    st.session_state[
        "prediction_case_message"
    ] = ""

    st.rerun()


# --------------------------------------------------
# Prediction input
# --------------------------------------------------

st.write(
    "### 📝 Transaction Input"
)


input_col1, input_col2, input_col3 = st.columns(3)


with input_col1:

    prediction_step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=1,
        step=1,
        key="prediction_step"
    )


with input_col2:

    prediction_type = st.selectbox(
        "Transaction Type",
        [
            "PAYMENT",
            "TRANSFER",
            "CASH_OUT",
            "DEBIT",
            "CASH_IN"
        ],
        key="prediction_type"
    )


with input_col3:

    prediction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        key="prediction_amount"
    )


input_col4, input_col5 = st.columns(2)


with input_col4:

    prediction_sender = st.text_input(
        "Sender Account",
        value="C123456789",
        key="prediction_sender"
    )


with input_col5:

    prediction_receiver = st.text_input(
        "Receiver Account",
        value="C987654321",
        key="prediction_receiver"
    )


balance_col1, balance_col2 = st.columns(2)


with balance_col1:

    prediction_old_sender = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        key="prediction_old_sender"
    )


with balance_col2:

    prediction_new_sender = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        key="prediction_new_sender"
    )


balance_col3, balance_col4 = st.columns(2)


with balance_col3:

    prediction_old_receiver = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        key="prediction_old_receiver"
    )


with balance_col4:

    prediction_new_receiver = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        key="prediction_new_receiver"
    )


count_col1, count_col2 = st.columns(2)


with count_col1:

    prediction_sender_count = st.number_input(
        "Sender Transaction Count",
        min_value=1,
        value=1,
        step=1,
        key="prediction_sender_count"
    )


with count_col2:

    prediction_receiver_count = st.number_input(
        "Receiver Transaction Count",
        min_value=1,
        value=1,
        step=1,
        key="prediction_receiver_count"
    )


# --------------------------------------------------
# Analyze transaction
# --------------------------------------------------

if st.button(
    "🔍 Analyze Transaction",
    key="analyze_new_transaction"
):

    transaction_data = {

        "step":
            int(prediction_step),

        "transaction_type":
            prediction_type,

        "amount":
            float(prediction_amount),

        "sender":
            prediction_sender,

        "receiver":
            prediction_receiver,

        "oldbalance_org":
            float(prediction_old_sender),

        "newbalance_orig":
            float(prediction_new_sender),

        "oldbalance_dest":
            float(prediction_old_receiver),

        "newbalance_dest":
            float(prediction_new_receiver),

        "sender_transaction_count":
            int(prediction_sender_count),

        "receiver_transaction_count":
            int(prediction_receiver_count)
    }


    try:

        prediction_result = (
            predict_transaction(
                transaction_data
            )
        )


        st.session_state[
            "prediction_result"
        ] = prediction_result


        st.session_state[
            "prediction_transaction"
        ] = transaction_data


        st.session_state[
            "prediction_case_created"
        ] = False


        st.session_state[
            "prediction_case_message"
        ] = ""


        st.success(
            "Transaction analyzed successfully."
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ==================================================
# Prediction Result
# ==================================================

prediction_result = st.session_state.get(
    "prediction_result"
)


prediction_transaction = st.session_state.get(
    "prediction_transaction"
)


if prediction_result is not None:

    st.write(
        "### 📊 Prediction Result"
    )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        st.metric(
            "Risk Score",
            f"{prediction_result['risk_score']:.2f}/100"
        )


    with result_col2:

        st.metric(
            "Fraud Probability",
            f"{prediction_result['fraud_probability'] * 100:.2f}%"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            prediction_result["risk_level"]
        )


    st.write(
        "### Prediction Details"
    )


    prediction_detail1, prediction_detail2, prediction_detail3 = (
        st.columns(3)
    )


    with prediction_detail1:

        st.metric(
            "Anomaly Risk",
            f"{prediction_result['anomaly_risk'] * 100:.2f}%"
        )


    with prediction_detail2:

        st.metric(
            "Rule Risk",
            f"{prediction_result['rule_risk'] * 100:.2f}%"
        )


    with prediction_detail3:

        st.metric(
            "Anomaly Prediction",
            prediction_result["anomaly_prediction"]
        )


    st.divider()


    # ==================================================
    # Create Investigation Case from Prediction
    # ==================================================

    st.write(
        "### 🗂️ Create Investigation Case"
    )


    predicted_status_col, predicted_priority_col = (
        st.columns(2)
    )


    with predicted_status_col:

        predicted_case_status = st.selectbox(
            "Case Status",
            [
                "Flagged",
                "Under Investigation",
                "Cleared"
            ],
            index=1,
            key="predicted_case_status"
        )


    with predicted_priority_col:

        prediction_level = (
            prediction_result["risk_level"]
        )


        if prediction_level == "CRITICAL":

            prediction_priority_index = 0

        elif prediction_level == "HIGH":

            prediction_priority_index = 1

        elif prediction_level == "MEDIUM":

            prediction_priority_index = 2

        else:

            prediction_priority_index = 3


        predicted_case_priority = st.selectbox(
            "Case Priority",
            [
                "Critical",
                "High",
                "Medium",
                "Low"
            ],
            index=prediction_priority_index,
            key="predicted_case_priority"
        )


    predicted_case_notes = st.text_area(
        "Investigator Notes",
        value=(
            "High-risk transaction detected by "
            "real-time prediction system."
        ),
        key="predicted_case_notes"
    )


    if st.button(
        "💾 Create Investigation Case",
        key="create_predicted_case"
    ):

        predicted_case_id = (
            prediction_transaction["sender"]
            + "_"
            + str(
                prediction_transaction["step"]
            )
        )


        predicted_case_data = {

            "transaction_id":
                predicted_case_id,

            "transaction_type":
                prediction_transaction[
                    "transaction_type"
                ],

            "amount":
                prediction_transaction[
                    "amount"
                ],

            "sender":
                prediction_transaction[
                    "sender"
                ],

            "receiver":
                prediction_transaction[
                    "receiver"
                ],

            "risk_score":
                prediction_result[
                    "risk_score"
                ],

            "risk_level":
                prediction_result[
                    "risk_level"
                ],

            "fraud_probability":
                prediction_result[
                    "fraud_probability"
                ],

            "case_status":
                predicted_case_status,

            "case_priority":
                predicted_case_priority,

            "investigator_notes":
                predicted_case_notes
        }


        try:

            response = create_case(
                predicted_case_data
            )


            load_cases.clear()


            st.session_state[
                "prediction_case_created"
            ] = True


            st.session_state[
                "prediction_case_message"
            ] = (
                "Investigation case created successfully: "
                f"{response['transaction_id']}"
            )


        except Exception as e:

            st.session_state[
                "prediction_case_created"
            ] = False


            st.session_state[
                "prediction_case_message"
            ] = (
                f"Failed to create investigation case: {e}"
            )


    if st.session_state[
        "prediction_case_message"
    ]:

        if st.session_state[
            "prediction_case_created"
        ]:

            st.success(
                st.session_state[
                    "prediction_case_message"
                ]
            )

        else:

            st.error(
                st.session_state[
                    "prediction_case_message"
                ]
            )


st.divider()

st.divider()

from src.model_evaluation import display_model_evaluation

display_model_evaluation()

st.divider()
# ==================================================
# Footer
# ==================================================

st.caption(
    "Fraud Investigation Platform | "
    "Random Forest + Isolation Forest + "
    "Rule-Based Risk Engine + Network Analysis + "
    "FastAPI + Streamlit"
)