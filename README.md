# 🔮 Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green?logo=githubactions)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25-brightgreen)
![AUC](https://img.shields.io/badge/AUC--ROC-0.91-brightgreen)

> **End-to-end ML pipeline** predicting customer churn with **87% accuracy** and **0.91 AUC-ROC** across 40+ engineered features including RFM metrics and behavioral signals. Deployed as a **REST API** on GCP Vertex AI with weekly automated CRM scoring.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CHURN PREDICTION PIPELINE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   CRM Data (10K customers)                                   │
│         │                                                    │
│         ▼                                                    │
│   ┌───────────────────────┐                                 │
│   │   Feature Engineering  │  40+ features                  │
│   │   ✓ RFM metrics        │  recency · frequency · monetary │
│   │   ✓ Behavioral signals │  NPS · login rate · complaints  │
│   │   ✓ Derived features   │  CLV · engagement score         │
│   └──────────┬────────────┘                                 │
│              │                                               │
│              ▼                                               │
│   ┌───────────────────────┐                                 │
│   │   XGBoost Classifier   │  300 estimators                │
│   │   + SHAP Explainer     │  Class balancing               │
│   │                        │  Feature importance per pred.  │
│   └──────────┬────────────┘                                 │
│              │                                               │
│              ▼                                               │
│   ┌───────────────────────┐                                 │
│   │   Model Evaluation     │  87% accuracy                  │
│   │                        │  0.91 AUC-ROC                  │
│   │                        │  Stratified K-Fold CV          │
│   └──────────┬────────────┘                                 │
│              │                                               │
│              ▼                                               │
│   ┌───────────────────────┐                                 │
│   │   FastAPI REST API     │  /predict                      │
│   │   (Dockerized)         │  /predict/batch                │
│   │                        │  /health · /metrics            │
│   └──────────┬────────────┘                                 │
│              │                                               │
│              ▼                                               │
│   ┌───────────────────────┐                                 │
│   │   GCP Vertex AI        │  Production deployment         │
│   │   CRM Integration      │  Weekly automated scoring      │
│   └───────────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | **87.2%** |
| **AUC-ROC** | **0.91** |
| **F1 Score** | 0.84 |
| **Precision** | 0.86 |
| **Recall** | 0.82 |

---

## 🔍 Top Features (SHAP)

```
1. recency_days             ████████████████████ 0.42
2. engagement_score         ███████████████      0.31
3. complaints               ████████████         0.24
4. rfm_score                ██████████           0.20
5. nps_score                █████████            0.18
6. clv_estimate             ████████             0.16
7. support_tickets          ███████              0.14
8. login_frequency_monthly  ██████               0.12
```

---

## 🛠️ Tech Stack

```
ML Model        → XGBoost 2.0
Explainability  → SHAP
Feature Eng.    → pandas · scikit-learn
API             → FastAPI + Uvicorn
Containerization→ Docker
CI/CD           → GitHub Actions
Cloud           → GCP Vertex AI
Testing         → pytest (20 tests)
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/NIKHILBODDAPATI/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install
pip install -r requirements.txt

# 3. Generate dataset
python data/generate_data.py
# → 10,000 customers · 40+ features · saved to data/customer_churn.csv

# 4. Train model
python model/train.py
# → Accuracy: 87% · AUC-ROC: 0.91 · artifacts saved

# 5. Start API
uvicorn api.api:app --reload --port 8000
# → Docs at http://localhost:8000/docs
```

---

## 🌐 API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "CUST_001", "recency_days": 280,
       "complaints": 2, "nps_score": 3, ...}'
```

```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.78,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "recommendation": "Immediate outreach — escalate to account manager"
}
```

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=data
# 20 tests · data quality · revenue logic · churn signal validation
```

---

## 📁 Structure

```
customer-churn-prediction/
├── data/generate_data.py      # 10K customers · 40+ features
├── model/train.py             # XGBoost + SHAP + evaluation
├── api/api.py                 # FastAPI · /predict · /batch
├── tests/test_pipeline.py     # 20 pytest tests
├── .github/workflows/ci.yml   # CI/CD · test + train + validate
├── Dockerfile
└── docker-compose.yml
```

---

**👤 Nikhil Boddapati** · [LinkedIn](https://linkedin.com/in/nikhil-boddapati) · [GitHub](https://github.com/NIKHILBODDAPATI)
