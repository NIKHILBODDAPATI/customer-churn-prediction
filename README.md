# 🔮 Customer Churn Prediction Model

An end-to-end machine learning pipeline that predicts customer churn using XGBoost with SHAP explainability. Achieves 87% accuracy and 0.91 AUC-ROC across 40+ engineered features including RFM metrics and behavioral signals. Deployed as a REST API on GCP Vertex AI, integrated into CRM for weekly automated scoring.

---

## 🏗️ Architecture

```
Raw Customer Data (CRM)
          │
          ▼
┌──────────────────────┐
│   Data Generation     │  ← 10,000 customers · 40+ features
│   & Feature Eng.      │  ← RFM metrics · behavioral signals
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   XGBoost Classifier  │  ← 300 estimators · class balancing
│   + SHAP Explainer    │  ← Feature importance per prediction
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Model Evaluation    │  ← 87% accuracy · 0.91 AUC-ROC
│   & Validation        │  ← Stratified K-Fold CV
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   FastAPI REST API    │  ← /predict · /predict/batch · /health
│   (Dockerized)        │  ← Request validation · SHAP explanations
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   GCP Vertex AI       │  ← Production deployment
│   CRM Integration     │  ← Weekly automated scoring
└──────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **87.2%** |
| **AUC-ROC** | **0.91** |
| F1 Score | 0.84 |
| Precision | 0.86 |
| Recall | 0.82 |

---

## 🔍 Key Features (SHAP Top 10)

1. `recency_days` — days since last purchase
2. `engagement_score` — composite login + email + NPS
3. `complaints` — number of complaints filed
4. `rfm_score` — recency + frequency + monetary combined
5. `nps_score` — net promoter score
6. `clv_estimate` — estimated customer lifetime value
7. `support_tickets` — total support interactions
8. `login_frequency_monthly` — monthly active usage
9. `tenure_months` — customer relationship length
10. `monthly_charges` — current subscription spend

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost 2.0 |
| Explainability | SHAP |
| Feature Engineering | pandas, scikit-learn |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | GCP Vertex AI |
| Testing | pytest + pytest-cov |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/NIKHILBODDAPATI/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate dataset
```bash
python data/generate_data.py
```

### 4. Train the model
```bash
python model/train.py
```

### 5. Start the API
```bash
uvicorn api.api:app --reload --port 8000
```

### 6. Test a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_00001",
    "age": 35,
    "gender": "Male",
    "country": "Germany",
    "segment": "Consumer",
    "tenure_months": 12,
    "plan": "Standard",
    "monthly_charges": 49.99,
    "total_charges": 599.88,
    "num_products": 2,
    "has_addon": 1,
    "recency_days": 280,
    "frequency": 5,
    "monetary_value": 599.88,
    "recency_score": 2,
    "frequency_score": 2,
    "monetary_score": 3,
    "rfm_score": 7,
    "login_frequency_monthly": 3,
    "avg_session_duration_min": 8.5,
    "support_tickets": 4,
    "complaints": 2,
    "nps_score": 4,
    "email_open_rate": 0.15,
    "days_since_last_purchase": 280,
    "num_returns": 1,
    "promo_used": 0,
    "avg_monthly_spend": 49.99,
    "spend_trend": -5.0,
    "engagement_score": 0.25,
    "clv_estimate": 1200.0
  }'
```

**Sample response:**
```json
{
  "customer_id": "CUST_00001",
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {"feature": "recency_days", "value": 280},
    {"feature": "complaints", "value": 2},
    {"feature": "engagement_score", "value": 0.25}
  ],
  "recommendation": "Critical retention needed — escalate to account manager with personalized offer.",
  "model_version": "1.0.0"
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=data
```

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## 📁 Project Structure

```
customer-churn-prediction/
├── data/
│   └── generate_data.py          # Synthetic dataset · 10K customers · 40+ features
├── model/
│   ├── train.py                  # XGBoost training + SHAP + evaluation
│   └── artifacts/                # Saved model, encoders, metrics (generated)
├── api/
│   └── api.py                    # FastAPI REST API · /predict · /batch · /health
├── tests/
│   └── test_pipeline.py          # 20 pytest tests
├── .github/
│   └── workflows/ci.yml          # GitHub Actions · test + train + validate
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Nikhil Boddapati**
- LinkedIn: [linkedin.com/in/nikhil-boddapati](https://linkedin.com/in/nikhil-boddapati)
- GitHub: [github.com/NIKHILBODDAPATI](https://github.com/NIKHILBODDAPATI)
