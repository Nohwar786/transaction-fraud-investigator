import pandas as pd
import networkx as nx


def build_transaction_graph(df):

    print("Building transaction graph...")

    graph = nx.DiGraph()

    # Only use meaningful money-transfer transactions
    transactions = df[
        df["type"].isin(["TRANSFER", "CASH_OUT"])
    ].copy()

    for _, row in transactions.iterrows():

        sender = row["nameOrig"]
        receiver = row["nameDest"]

        graph.add_edge(
            sender,
            receiver,
            amount=row["amount"],
            risk_score=row["risk_score"],
            isFraud=row["isFraud"]
        )

    return graph


def calculate_network_features(graph):

    print("Calculating network features...")

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    total_degree = {
        node: in_degree.get(node, 0)
        + out_degree.get(node, 0)
        for node in graph.nodes()
    }

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree
    }


if __name__ == "__main__":

    print("Loading investigation dataset...")

    df = pd.read_csv(
        "data/investigation_transactions.csv"
    )

    # To keep graph construction manageable,
    # focus on suspicious transactions.
    suspicious = df[
        df["risk_score"] >= 60
    ].copy()

    print(
        f"Suspicious transactions: "
        f"{len(suspicious):,}"
    )

    graph = build_transaction_graph(
        suspicious
    )

    print(
        f"\nGraph nodes: {graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: {graph.number_of_edges():,}"
    )

    network_features = calculate_network_features(
        graph
    )

    suspicious["sender_network_degree"] = (
        suspicious["nameOrig"]
        .map(network_features["total_degree"])
        .fillna(0)
    )

    suspicious["receiver_network_degree"] = (
        suspicious["nameDest"]
        .map(network_features["total_degree"])
        .fillna(0)
    )

    # Accounts involved in highly connected networks
    suspicious["network_risk"] = (
        suspicious["sender_network_degree"]
        + suspicious["receiver_network_degree"]
    )

    print("\nMost connected suspicious accounts:")

    print(
        suspicious[
            [
                "nameOrig",
                "nameDest",
                "amount",
                "risk_score",
                "sender_network_degree",
                "receiver_network_degree",
                "network_risk",
                "isFraud"
            ]
        ]
        .sort_values(
            "network_risk",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    suspicious.to_csv(
        "data/network_investigation.csv",
        index=False
    )

    print(
        "\nNetwork investigation dataset saved to "
        "data/network_investigation.csv"
    )