import requests


API_URL = "http://127.0.0.1:8000"


def get_health():
    response = requests.get(
        f"{API_URL}/health"
    )

    response.raise_for_status()

    return response.json()


def get_alerts():
    response = requests.get(
        f"{API_URL}/alerts"
    )

    response.raise_for_status()

    return response.json()


def get_cases():
    response = requests.get(
        f"{API_URL}/cases"
    )

    response.raise_for_status()

    return response.json()


def predict_transaction(transaction):
    response = requests.post(
        f"{API_URL}/predict",
        json=transaction
    )

    response.raise_for_status()

    return response.json()


def create_case(case):
    response = requests.post(
        f"{API_URL}/cases",
        json=case
    )

    response.raise_for_status()

    return response.json()