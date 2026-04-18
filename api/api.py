"""
Customer Churn Prediction - REST API
FastAPI endpoint for real-time churn scoring
Deployed on GCP Vertex AI / local Docker
"""

import pickle
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import os

# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_DIR = "model/artifacts"

def load_artifacts():
    try:
        with open(f"{MODEL_DIR}/xgboost_churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{MODEL_DIR}/encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(f"{MODEL_DIR}/metrics.json", "r") as f:
            metrics = json.load(f)
        return model, encoders, metrics
    except FileNotFoundError:
        raise RuntimeError("Model artifacts not found. Run model/train.py first.")

model, encoders, model_metrics = load_artifacts()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn scoring using XGBoost + SHAP explainability",
    version="1.0.0",
)

# ── Request schema ────────────────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id: str
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., pattern="^(Male|Female)$")
    country: str
    segment: str
    tenure_months: int = Field(..., ge=0)
    plan: str = Field(..., pattern="^(Basic|Standard|Premium)$")
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    has_addon: int = Field(..., ge=0, le=1)
    recency_days: int = Field(..., ge=0)
    frequency: int = Field(..., ge=0)
    monetary_value: float = Field(..., ge=0)
    recency_score: int = Field(..., ge=1, le=5)
    frequency_score: int = Field(..., ge=1, le=5)
    monetary_score: int = Field(..., ge=1, le=5)
    rfm_score: int = Field(..., ge=3, le=15)
    login_frequency_monthly: int = Field(..., ge=0)
    avg_session_duration_min: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    complaints: int = Field(..., ge=0)
    nps_score: int = Field(..., ge=0, le=10)
    email_open_rate: float = Field(..., ge=0, le=1)
    days_since_last_purchase: int = Field(..., ge=0)
    num_returns: int = Field(..., ge=0)
    promo_used: int = Field(..., ge=0, le=1)
    avg_monthly_spend: float = Field(..., ge=0)
    spend_trend: float
    engagement_score: float = Field(..., ge=0, le=1)
    clv_estimate: float = Field(..., ge=0)

# ── Response schema ───────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    top_risk_factors: list
    recommendation: str
    model_version: str = "1.0.0"


def engineer_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering as training."""
    df = pd.DataFrame([data])

    df["revenue_per_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
    df["tickets_per_month"] = df["support_tickets"] / df["tenure_months"].clip(lower=1)
    df["complaints_ratio"] = df["complaints"] / (df["support_tickets"] + 1)
    df["engagement_x_rfm"] = df["engagement_score"] * df["rfm_score"]
    df["is_high_value"] = (df["clv_estimate"] > 2400).astype(int)
    df["is_long_tenure"] = (df["tenure_months"] > 24).astype(int)
    df["spend_per_product"] = df["monthly_charges"] / df["num_products"].clip(lower=1)
    df["recency_x_frequency"] = df["recency_days"] * df["frequency"]

    # Encode categoricals
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError:
                df[col] = 0  # unseen category

    return df.drop(columns=["customer_id"], errors="ignore")


def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def get_recommendation(prob: float, features: dict) -> str:
    if prob >= 0.7:
        if features.get("complaints", 0) > 2:
            return "Immediate outreach — customer has multiple complaints. Offer dedicated support and service credit."
        elif features.get("nps_score", 5) < 5:
            return "High churn risk — schedule customer success call and offer loyalty discount."
        return "Critical retention needed — escalate to account manager with personalized offer."
    elif prob >= 0.4:
        if features.get("login_frequency_monthly", 10) < 5:
            return "Low engagement detected — trigger re-engagement email campaign with product tips."
        return "Moderate risk — enroll in loyalty program and send personalized recommendations."
    return "Customer is healthy — continue standard engagement. Consider upsell opportunity."


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "model_metrics": model_metrics,
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    """Score a single customer for churn probability."""
    try:
        data = customer.dict()
        customer_id = data.pop("customer_id") if "customer_id" in data else "unknown"
        data["customer_id"] = customer_id

        # Feature engineering
        X = engineer_features(data)

        # Predict
        churn_prob = float(model.predict_proba(X)[0][1])
        churn_pred = churn_prob >= 0.5

        # Top risk factors (feature importance proxy)
        feature_scores = dict(zip(X.columns, X.values[0]))
        top_risk = sorted(
            [{"feature": k, "value": round(float(v), 3)}
             for k, v in feature_scores.items()
             if k in ["recency_days", "complaints", "support_tickets",
                      "nps_score", "login_frequency_monthly", "engagement_score"]],
            key=lambda x: abs(x["value"]),
            reverse=True
        )[:3]

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(churn_prob, 4),
            churn_prediction=churn_pred,
            risk_level=get_risk_level(churn_prob),
            top_risk_factors=top_risk,
            recommendation=get_recommendation(churn_prob, data),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerFeatures]):
    """Score multiple customers in one request (weekly CRM batch)."""
    return [predict(c) for c in customers]


@app.get("/model/metrics")
def get_metrics():
    """Return model performance metrics."""
    return model_metrics


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
