from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# LOAD MODEL
model = joblib.load("fraud_detection_model.pkl")

# DEFINE FEATURES
numeric_features = [
    "amount_ngn", "spending_deviation_score", "velocity_score",
    "geo_anomaly_score", "device_seen_count", "ip_seen_count",
    "user_txn_count_total", "user_txn_count_sample", "user_avg_amt_sample",
    "user_std_amt_sample", "avg_gap_between_txns_sample",
    "time_since_last_sample", "merchant_fraud_rate_sample",
    "channel_risk_score_sample", "persona_fraud_risk_sample",
    "location_fraud_risk_sample", "device_fraud_rate_sample",
    "ip_fraud_rate_sample", "txn_count_last_1h_sample",
    "txn_count_last_24h_sample", "amount_risk_merchant",
    "amount_risk_channel", "amount_risk_persona", "amount_risk_location",
    "device_persona_risk", "ip_persona_risk", "burst_activity_score",
    "txn_hour_sin", "txn_hour_cos", "deviation_from_user_avg",
    "overall_risk_score"
]

categorical_features = [
    "transaction_type", "merchant_category", "location", "device_used",
    "payment_channel", "sender_persona", "user_top_category",
    "ip_geo_region"
]

binary_features = [
    "bvn_linked", "new_device_transaction", "geospatial_velocity_anomaly",
    "is_weekend", "is_salary_week", "is_night_txn",
    "is_device_shared", "is_ip_shared"
]

all_features = numeric_features + categorical_features + binary_features

# PYDANTIC MODEL
class Transaction(BaseModel):
    amount_ngn: float
    spending_deviation_score: float
    velocity_score: float
    geo_anomaly_score: float
    device_seen_count: int
    ip_seen_count: int
    user_txn_count_total: int
    user_txn_count_sample: int
    user_avg_amt_sample: float
    user_std_amt_sample: float
    avg_gap_between_txns_sample: float
    time_since_last_sample: float
    merchant_fraud_rate_sample: float
    channel_risk_score_sample: float
    persona_fraud_risk_sample: float
    location_fraud_risk_sample: float
    device_fraud_rate_sample: float
    ip_fraud_rate_sample: float
    txn_count_last_1h_sample: int
    txn_count_last_24h_sample: int
    amount_risk_merchant: float
    amount_risk_channel: float
    amount_risk_persona: float
    amount_risk_location: float
    device_persona_risk: float
    ip_persona_risk: float
    burst_activity_score: float
    txn_hour_sin: float
    txn_hour_cos: float
    deviation_from_user_avg: float
    overall_risk_score: float

    transaction_type: str
    merchant_category: str
    location: str
    device_used: str
    payment_channel: str
    sender_persona: str
    user_top_category: str
    ip_geo_region: str

    bvn_linked: int
    new_device_transaction: int
    geospatial_velocity_anomaly: int
    is_weekend: int
    is_salary_week: int
    is_night_txn: int
    is_device_shared: int
    is_ip_shared: int

# Utility: Clean Numpy types
def clean(value):
    if isinstance(value, (np.generic,)):
        return value.item()
    return value

# SCORING FUNCTION
def score_transaction(transaction: dict):
    df = pd.DataFrame([transaction])[all_features]

    df[numeric_features] = df[numeric_features].astype(float)
    df[binary_features] = df[binary_features].astype(int)
    df[categorical_features] = df[categorical_features].astype(str)

    prob = float(model.predict_proba(df)[0][1])

    if prob >= 0.40:
        risk = "Critical Fraud Risk"
    elif prob >= 0.20:
        risk = "High Fraud Risk"
    elif prob >= 0.06:
        risk = "Suspicious Activity"
    elif prob >= 0.03:
        risk = "Medium Risk Anomaly"
    else:
        risk = "Normal Activity"

    return {
        "probability": clean(prob),
        "risk": clean(risk)
    }

# FASTAPI APP
app = FastAPI(title="FinSecure Fraud Detection API")

@app.get("/")
def home():
    return {"message": "FinSecure API running ✔"}

# SINGLE ITEM ENDPOINT
@app.post("/predict")
def predict(transaction: Transaction):
    return score_transaction(transaction.model_dump())

# MULTIPLE ITEMS ENDPOINT
@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    results = []
    for tx in transactions:
        results.append(score_transaction(tx.model_dump()))
    return {"results": results}
