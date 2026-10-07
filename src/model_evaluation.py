import os
import json
import pandas as pd
import streamlit as st


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


EVALUATION_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "model_evaluation.json"
)


FEATURE_IMPORTANCE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "feature_importance.csv"
)


def load_model_evaluation():

    if not os.path.exists(EVALUATION_PATH):
        return None

    with open(
        EVALUATION_PATH,
        "r"
    ) as file:

        return json.load(file)


def load_feature_importance():

    if not os.path.exists(
        FEATURE_IMPORTANCE_PATH
    ):
        return pd.DataFrame()

    return pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )


def display_model_evaluation():

    evaluation = load_model_evaluation()

    feature_importance = load_feature_importance()

    st.subheader(
        "🤖 Model Evaluation"
    )

    if evaluation is None:

        st.warning(
            "Model evaluation results are not available."
        )

        return

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    metric1, metric2, metric3, metric4, metric5 = (
        st.columns(5)
    )

    with metric1:

        st.metric(
            "Precision",
            f"{evaluation['precision'] * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Recall",
            f"{evaluation['recall'] * 100:.2f}%"
        )

    with metric3:

        st.metric(
            "F1 Score",
            f"{evaluation['f1_score'] * 100:.2f}%"
        )

    with metric4:

        st.metric(
            "ROC-AUC",
            f"{evaluation['roc_auc']:.4f}"
        )

    with metric5:

        st.metric(
            "PR-AUC",
            f"{evaluation['pr_auc']:.4f}"
        )

    st.write(
        f"Test samples: **{evaluation['test_samples']:,}**"
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    st.write(
        "### Confusion Matrix"
    )

    cm = evaluation["confusion_matrix"]

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Fraud"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Fraud"
        ]
    )

    st.dataframe(
        confusion_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    if not feature_importance.empty:

        st.write(
            "### Top Fraud Detection Features"
        )

        top_features = (
            feature_importance
            .head(10)
            .copy()
        )

        top_features = (
            top_features
            .set_index("feature")
        )

        st.bar_chart(
            top_features["importance"]
        )

        st.dataframe(
            feature_importance.head(10),
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Evaluation is based on the held-out test set. "
        "The PaySim dataset is synthetic, so these metrics "
        "should not be interpreted as real-world fraud "
        "detection performance."
    )