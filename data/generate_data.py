"""
Customer Churn Prediction - Data Generator
Generates synthetic customer dataset with RFM metrics and behavioral signals
40+ features · ~10,000 customers
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)
N_CUSTOMERS = 10000


def generate_churn_dataset() -> pd.DataFrame:
    """Generate synthetic customer dataset with 40+ features."""

    print("🔄 Generating synthetic customer dataset...")

    # ── Basic customer info ───────────────────────────────────────────────────
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
    tenure_months = np.random.randint(1, 72, N_CUSTOMERS)
    age = np.random.randint(18, 75, N_CUSTOMERS)
    gender = np.random.choice(["Male", "Female"], N_CUSTOMERS)
    country = np.random.choice(
        ["Germany", "France", "UK", "Netherlands", "Spain"],
        N_CUSTOMERS, p=[0.4, 0.2, 0.2, 0.1, 0.1]
    )
    segment = np.random.choice(
        ["Consumer", "Corporate", "Home Office"],
        N_CUSTOMERS, p=[0.6, 0.3, 0.1]
    )

    # ── RFM Metrics ───────────────────────────────────────────────────────────
    recency_days = np.random.randint(1, 365, N_CUSTOMERS)
    frequency = np.random.randint(1, 50, N_CUSTOMERS)
    monetary_value = np.round(np.random.exponential(200, N_CUSTOMERS), 2)

    # RFM scores (1-5)
    recency_score = pd.qcut(recency_days, 5, labels=[5, 4, 3, 2, 1]).astype(int)
    frequency_score = pd.qcut(frequency.clip(1, 49), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    monetary_score = pd.qcut(monetary_value.clip(1, None), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    rfm_score = recency_score + frequency_score + monetary_score

    # ── Product / Subscription ────────────────────────────────────────────────
    plan = np.random.choice(["Basic", "Standard", "Premium"], N_CUSTOMERS, p=[0.3, 0.5, 0.2])
    monthly_charges = np.where(
        plan == "Basic", np.random.uniform(10, 30, N_CUSTOMERS),
        np.where(plan == "Standard", np.random.uniform(30, 70, N_CUSTOMERS),
                 np.random.uniform(70, 150, N_CUSTOMERS))
    ).round(2)
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.8, 1.0, N_CUSTOMERS)).round(2)
    num_products = np.random.randint(1, 6, N_CUSTOMERS)
    has_addon = np.random.choice([0, 1], N_CUSTOMERS, p=[0.4, 0.6])

    # ── Behavioral signals ────────────────────────────────────────────────────
    login_frequency_monthly = np.random.randint(0, 30, N_CUSTOMERS)
    avg_session_duration_min = np.round(np.random.exponential(15, N_CUSTOMERS), 2)
    support_tickets = np.random.randint(0, 10, N_CUSTOMERS)
    complaints = np.random.randint(0, 5, N_CUSTOMERS)
    nps_score = np.random.randint(0, 11, N_CUSTOMERS)
    email_open_rate = np.round(np.random.uniform(0, 1, N_CUSTOMERS), 3)
    days_since_last_purchase = np.random.randint(1, 400, N_CUSTOMERS)
    num_returns = np.random.randint(0, 8, N_CUSTOMERS)
    promo_used = np.random.choice([0, 1], N_CUSTOMERS, p=[0.5, 0.5])

    # ── Derived features ──────────────────────────────────────────────────────
    avg_monthly_spend = (total_charges / tenure_months.clip(1)).round(2)
    spend_trend = np.round(np.random.normal(0, 20, N_CUSTOMERS), 2)  # positive = increasing
    engagement_score = (
        (login_frequency_monthly / 30) * 0.4 +
        (email_open_rate) * 0.3 +
        ((10 - nps_score) / 10) * 0.3
    ).round(3)
    clv_estimate = (monthly_charges * 12 * np.random.uniform(0.5, 3.0, N_CUSTOMERS)).round(2)

    # ── Churn label (target) ──────────────────────────────────────────────────
    # Churn probability influenced by key signals
    churn_prob = (
        0.3 * (recency_days / 365) +
        0.2 * (1 - login_frequency_monthly / 30) +
        0.2 * (complaints / 5) +
        0.15 * (support_tickets / 10) +
        0.1 * (1 - email_open_rate) +
        0.05 * (1 - rfm_score / 15)
    ).clip(0, 1)

    churn = (np.random.uniform(0, 1, N_CUSTOMERS) < churn_prob).astype(int)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "gender": gender,
        "country": country,
        "segment": segment,
        "tenure_months": tenure_months,
        "plan": plan,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "has_addon": has_addon,
        "recency_days": recency_days,
        "frequency": frequency,
        "monetary_value": monetary_value,
        "recency_score": recency_score,
        "frequency_score": frequency_score,
        "monetary_score": monetary_score,
        "rfm_score": rfm_score,
        "login_frequency_monthly": login_frequency_monthly,
        "avg_session_duration_min": avg_session_duration_min,
        "support_tickets": support_tickets,
        "complaints": complaints,
        "nps_score": nps_score,
        "email_open_rate": email_open_rate,
        "days_since_last_purchase": days_since_last_purchase,
        "num_returns": num_returns,
        "promo_used": promo_used,
        "avg_monthly_spend": avg_monthly_spend,
        "spend_trend": spend_trend,
        "engagement_score": engagement_score,
        "clv_estimate": clv_estimate,
        "churn": churn,
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customer_churn.csv", index=False)

    churn_rate = churn.mean() * 100
    print(f"✅ Dataset generated: {len(df):,} customers")
    print(f"   Churn rate: {churn_rate:.1f}%")
    print(f"   Features: {len(df.columns) - 2} (excl. customer_id and target)")
    print(f"   Saved to: data/customer_churn.csv")
    return df


if __name__ == "__main__":
    generate_churn_dataset()
