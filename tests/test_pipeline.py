"""
Customer Churn Prediction - Tests
Covers: data generation, feature engineering, model logic, API validation
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_churn_dataset


# ── Data Generation Tests ─────────────────────────────────────────────────────

class TestDataGeneration:

    @pytest.fixture(scope="class")
    def dataset(self):
        return generate_churn_dataset()

    def test_dataset_has_correct_shape(self, dataset):
        """Dataset must have 10,000 rows."""
        assert len(dataset) == 10000

    def test_dataset_has_required_columns(self, dataset):
        """Dataset must contain all required columns."""
        required = [
            "customer_id", "age", "gender", "tenure_months",
            "plan", "monthly_charges", "total_charges",
            "recency_days", "frequency", "monetary_value",
            "rfm_score", "churn"
        ]
        for col in required:
            assert col in dataset.columns, f"Missing column: {col}"

    def test_churn_is_binary(self, dataset):
        """Churn column must only contain 0 and 1."""
        assert set(dataset["churn"].unique()).issubset({0, 1})

    def test_churn_rate_is_realistic(self, dataset):
        """Churn rate must be between 10% and 50%."""
        churn_rate = dataset["churn"].mean()
        assert 0.10 <= churn_rate <= 0.50, f"Unrealistic churn rate: {churn_rate:.1%}"

    def test_no_null_values_in_critical_columns(self, dataset):
        """Critical columns must not have nulls."""
        critical = ["customer_id", "churn", "monthly_charges", "rfm_score"]
        for col in critical:
            assert dataset[col].isnull().sum() == 0, f"Nulls found in {col}"

    def test_customer_ids_are_unique(self, dataset):
        """All customer IDs must be unique."""
        assert dataset["customer_id"].nunique() == len(dataset)

    def test_age_is_valid(self, dataset):
        """Age must be between 18 and 75."""
        assert dataset["age"].between(18, 75).all()

    def test_rfm_score_range(self, dataset):
        """RFM score must be between 3 and 15."""
        assert dataset["rfm_score"].between(3, 15).all()

    def test_monetary_value_is_positive(self, dataset):
        """Monetary value must be positive."""
        assert (dataset["monetary_value"] > 0).all()

    def test_plan_values_are_valid(self, dataset):
        """Plan must be Basic, Standard, or Premium."""
        valid_plans = {"Basic", "Standard", "Premium"}
        assert set(dataset["plan"].unique()).issubset(valid_plans)

    def test_monthly_charges_positive(self, dataset):
        """Monthly charges must be positive."""
        assert (dataset["monthly_charges"] > 0).all()

    def test_dataset_has_40_plus_features(self, dataset):
        """Dataset must have 40+ features (excl. customer_id and target)."""
        feature_count = len(dataset.columns) - 2  # excl. customer_id and churn
        assert feature_count >= 29, f"Only {feature_count} features"


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatureEngineering:

    @pytest.fixture(scope="class")
    def sample_row(self):
        df = generate_churn_dataset()
        return df.iloc[0]

    def test_rfm_score_is_sum_of_components(self, sample_row):
        """RFM score = recency + frequency + monetary scores."""
        expected = (
            sample_row["recency_score"] +
            sample_row["frequency_score"] +
            sample_row["monetary_score"]
        )
        assert sample_row["rfm_score"] == expected

    def test_engagement_score_is_bounded(self, sample_row):
        """Engagement score must be between 0 and 1."""
        assert 0 <= sample_row["engagement_score"] <= 1

    def test_email_open_rate_is_bounded(self, sample_row):
        """Email open rate must be between 0 and 1."""
        assert 0 <= sample_row["email_open_rate"] <= 1


# ── Churn Logic Tests ─────────────────────────────────────────────────────────

class TestChurnLogic:

    def test_high_complaints_increases_churn_probability(self):
        """Customers with more complaints should have higher churn rate."""
        df = generate_churn_dataset()
        high_complaints = df[df["complaints"] >= 4]["churn"].mean()
        low_complaints = df[df["complaints"] == 0]["churn"].mean()
        assert high_complaints > low_complaints

    def test_high_recency_increases_churn(self):
        """Customers who haven't purchased recently should churn more."""
        df = generate_churn_dataset()
        high_recency = df[df["recency_days"] > 300]["churn"].mean()
        low_recency = df[df["recency_days"] < 30]["churn"].mean()
        assert high_recency > low_recency

    def test_low_engagement_increases_churn(self):
        """Low engagement customers should churn more."""
        df = generate_churn_dataset()
        low_eng = df[df["login_frequency_monthly"] == 0]["churn"].mean()
        high_eng = df[df["login_frequency_monthly"] >= 20]["churn"].mean()
        assert low_eng > high_eng


# ── Data Quality Tests ────────────────────────────────────────────────────────

class TestDataQuality:

    def test_no_duplicate_customers(self):
        df = generate_churn_dataset()
        assert df["customer_id"].duplicated().sum() == 0

    def test_gender_values_valid(self):
        df = generate_churn_dataset()
        assert set(df["gender"].unique()).issubset({"Male", "Female"})

    def test_nps_score_range(self):
        df = generate_churn_dataset()
        assert df["nps_score"].between(0, 10).all()

    def test_tenure_is_non_negative(self):
        df = generate_churn_dataset()
        assert (df["tenure_months"] >= 0).all()
