# Transaction Fraud Investigator

An advanced multi-layer fraud investigation platform that combines machine learning, anomaly detection, rule-based risk scoring, explainability, network analysis, FastAPI, and Streamlit to investigate suspicious financial transactions.

## 🚀 Features

* Random Forest fraud classification
* Isolation Forest anomaly detection
* Rule-based risk engine
* Combined transaction risk score
* Explainable fraud investigation reasons
* Suspicious transaction alerts
* Account/network analysis using NetworkX
* Interactive suspicious network graph
* Investigation case management
* Real-time transaction prediction through FastAPI
* Interactive Streamlit investigation dashboard
* Model evaluation and feature importance
* Confusion matrix and ROC-AUC / PR-AUC evaluation

## 🏗️ Architecture

```text
Transactions
     ↓
Data Loading
     ↓
Feature Engineering
     ↓
 ┌───────────────────────┐
 │ Random Forest Model   │
 └───────────────────────┘
     ↓
 ┌───────────────────────┐
 │ Isolation Forest      │
 └───────────────────────┘
     ↓
 ┌───────────────────────┐
 │ Rule-Based Risk Engine│
 └───────────────────────┘
     ↓
Combined Risk Score
     ↓
Explainability
     ↓
Network Analysis
     ↓
Alerts & Case Management
     ↓
FastAPI + Streamlit Dashboard
```

## 📊 Dataset

This project uses the **PaySim synthetic financial transaction dataset**.

The dataset contains approximately 6.36 million transactions and includes transaction information such as:

* Transaction type
* Transaction amount
* Sender and receiver accounts
* Sender balances
* Receiver balances
* Fraud labels

The original dataset is **not included in this repository** because of its large file size.

## 🤖 Machine Learning

### Random Forest

The Random Forest classifier is used to estimate the probability that a transaction is fraudulent.

### Isolation Forest

Isolation Forest identifies transactions that are unusual compared with normal transaction behavior.

### Rule-Based Risk Engine

Additional investigation signals are generated from transaction behavior, including:

* Night-time transactions
* Large transactions
* Sender balance reaching zero
* Transfer transactions
* High transaction frequency
* Balance inconsistencies

### Combined Risk Score

The platform combines:

* Fraud probability
* Anomaly risk
* Rule-based risk

to produce a risk score from **0 to 100**.

Risk levels:

|  Score | Risk Level |
| -----: | ---------- |
|   0–29 | LOW        |
|  30–59 | MEDIUM     |
|  60–79 | HIGH       |
| 80–100 | CRITICAL   |

## 📈 Model Evaluation

The Random Forest model was evaluated on a held-out test set.

| Metric    | Result |
| --------- | -----: |
| Precision | 99.94% |
| Recall    | 99.76% |
| F1 Score  | 99.85% |
| ROC-AUC   | 0.9996 |
| PR-AUC    | 0.9977 |

These results are based on the synthetic PaySim dataset and should not be interpreted as real-world fraud detection performance.

## 🌐 Network Analysis

Suspicious transactions are analyzed as a transaction network.

The platform calculates:

* Sender network degree
* Receiver network degree
* Total network degree
* Network risk

This helps investigators identify suspicious connections and transaction relationships.

## 🖥️ Dashboard

The Streamlit dashboard provides:

* Investigation summary
* Risk distribution
* Suspicious activity over time
* Top suspicious accounts
* Transaction investigation
* Network investigation
* Suspicious network graph
* Investigation reasons
* Model assessment
* Case management
* Case history
* Real-time transaction prediction
* Model evaluation

## ⚡ FastAPI

The platform provides a REST API for real-time fraud risk assessment.

Main endpoints:

```text
GET  /
GET  /health
GET  /alerts
GET  /cases
POST /cases
POST /predict
```

## 📁 Project Structure

```text
transaction-fraud-investigator/
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── investigation_dashboard.py
│
├── data/
│   └── model_evaluation.json
│
├── src/
│   ├── anomaly_detector.py
│   ├── create_alerts.py
│   ├── data_loader.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── fraud_model.py
│   ├── graph_analysis.py
│   ├── model_evaluation.py
│   └── risk_engine.py
│
├── api_client.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Nohwar786/transaction-fraud-investigator.git
cd transaction-fraud-investigator
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Start the FastAPI backend

```bash
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Start the Streamlit dashboard

Open another terminal:

```bash
streamlit run dashboard/investigation_dashboard.py
```

The dashboard will open in your browser.

## 🔍 Real-Time Prediction

Transactions can be submitted through the FastAPI `/predict` endpoint.

The API returns:

* Fraud probability
* Anomaly risk
* Rule-based risk
* Combined risk score
* Risk level
* Anomaly prediction

## 🛡️ Data & Model Files

Large and generated files are intentionally excluded from GitHub through `.gitignore`, including:

* Original PaySim transaction dataset
* Generated transaction CSV files
* Trained model files
* Virtual environment
* Python cache files

This keeps the repository lightweight and focused on the source code.

## 🧰 Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* FastAPI
* Uvicorn
* NetworkX
* Joblib
* Requests

## 👨‍💻 Project Purpose

This project demonstrates how multiple fraud detection techniques can be combined into an investigation-oriented platform rather than relying only on a single machine learning classifier.

It is designed as a portfolio project demonstrating machine learning, anomaly detection, explainable investigation, graph analysis, REST APIs, and interactive dashboards.



## 📸 Project Screenshots

### 1. Dashboard

![Dashboard](screenshots/dashboard.png)

### 2. Transaction Data

![Transaction Data](screenshots/transaction-data.png)

### 3. Fraud Detection Results

![Fraud Detection](screenshots/fraud-detection.png)

### 4. Risk Analysis

![Risk Analysis](screenshots/risk-analysis.png)

### 5. Transaction Investigation

![Transaction Investigation](screenshots/transaction-investigation.png)

### 6. Anomaly Detection

![Anomaly Detection](screenshots/anomaly-detection.png)

### 7. Fraud Summary

![Fraud Summary](screenshots/fraud-summary.png)

### 8. Investigation Details

![Investigation Details](screenshots/investigation-details.png)

### 9. Additional Results

![Additional Results](screenshots/additional-results.png)
