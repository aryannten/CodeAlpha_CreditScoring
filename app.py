"""Streamlit Web Application: Credit Scoring & Default Risk Predictor.

Run locally via:
    streamlit run app.py

Features:
  - Dynamically adapts form inputs to whichever dataset the serialized model was trained on
    (German Credit dataset OR Give Me Some Credit dataset).
  - Real-time default risk probability calculation and risk categorization.
  - Interactive decision threshold tuning slider.
  - Educational disclaimer & CodeAlpha ML Internship attribution.
"""

from __future__ import annotations

import os
import streamlit as st

from src.predict import score_applicant, _load_model, _get_expected_features, DEFAULT_DECISION_THRESHOLD

st.set_page_config(
    page_title="Credit Scoring System | CodeAlpha",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">💳 Credit Scoring & Default Risk Assessment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predicting creditworthiness and probability of loan default '
    'using machine learning classification pipelines. '
    '<strong>Task 1 — CodeAlpha Machine Learning Internship</strong></div>',
    unsafe_allow_html=True,
)

# Load Model & Detect Dataset Schema
model_path = os.path.join("models", "best_model.joblib")
model_loaded = False
is_gmsc_schema = False
expected_features = []

if os.path.exists(model_path):
    try:
        pipeline = _load_model(model_path)
        expected_features = _get_expected_features(pipeline)
        model_loaded = True
        # Check if the model was trained on Give Me Some Credit
        if any("RevolvingUtilization" in f or "DebtRatio" in f for f in expected_features):
            is_gmsc_schema = True
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Sidebar
with st.sidebar:
    st.header("📌 System Parameters")
    dataset_label = "Give Me Some Credit (Kaggle)" if is_gmsc_schema else "German Credit Data (UCI / OpenML)"
    st.info(f"**Active Model Schema:**\n{dataset_label}")
    
    st.markdown(
        """
        **Risk Tier Definitions:**
        - 🟢 **Low Risk**: Default Prob < 20% (Approve)
        - 🟡 **Medium Risk**: 20% ≤ Default Prob < Threshold (Manual Review)
        - 🔴 **High Risk**: Default Prob ≥ Threshold (Decline / Collateral)
        """
    )
    st.divider()
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10,
        max_value=0.80,
        value=float(DEFAULT_DECISION_THRESHOLD),
        step=0.05,
        help="Probability cutoff for High-Risk classification.",
    )
    st.divider()
    st.caption("⚠️ **Educational Project** — Not financial advice.")

# Form Input
with st.form("credit_application_form"):
    st.subheader(f"📋 Applicant Profile ({dataset_label})")
    applicant_data = {}

    if is_gmsc_schema:
        # Give Me Some Credit Form
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_data["age"] = st.number_input("Applicant Age", min_value=18, max_value=100, value=38)
            applicant_data["MonthlyIncome"] = st.number_input("Monthly Income ($)", min_value=0, max_value=200000, value=5500, step=250)
            applicant_data["NumberOfDependents"] = st.number_input("Number of Dependents", min_value=0, max_value=15, value=1)
        with c2:
            applicant_data["DebtRatio"] = st.number_input("Debt Ratio (Monthly debt / Monthly income)", min_value=0.0, max_value=20.0, value=0.35, step=0.05)
            applicant_data["RevolvingUtilizationOfUnsecuredLines"] = st.slider("Revolving Line Utilization", min_value=0.0, max_value=1.5, value=0.25, step=0.05)
            applicant_data["NumberOfOpenCreditLinesAndLoans"] = st.number_input("Open Credit Lines & Loans", min_value=0, max_value=50, value=6)
        with c3:
            applicant_data["NumberOfTimes30-59DaysPastDueNotWorse"] = st.number_input("30-59 Days Past Due Count", min_value=0, max_value=20, value=0)
            applicant_data["NumberOfTimes60-89DaysPastDueNotWorse"] = st.number_input("60-89 Days Past Due Count", min_value=0, max_value=20, value=0)
            applicant_data["NumberOfTimes90DaysLate"] = st.number_input("90+ Days Late Count", min_value=0, max_value=20, value=0)
            applicant_data["NumberRealEstateLoansOrLines"] = st.number_input("Real Estate Loans or Lines", min_value=0, max_value=20, value=1)

    else:
        # German Credit Form
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Financial Liquidity**")
            applicant_data["checking_status"] = st.selectbox("Checking Account Status", ["<0", "0<=X<200", ">=200", "no checking"])
            applicant_data["savings_status"] = st.selectbox("Savings Account Balance", ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no savings"])
            applicant_data["credit_amount"] = st.number_input("Credit Amount Requested", min_value=250, max_value=25000, value=3000, step=250)
            applicant_data["duration"] = st.number_input("Loan Duration (Months)", min_value=4, max_value=72, value=24, step=1)
        with c2:
            st.markdown("**Credit History & Obligation**")
            applicant_data["credit_history"] = st.selectbox("Credit History", ["critical", "delayed", "existing paid", "all paid", "no credits/paid"])
            applicant_data["purpose"] = st.selectbox("Loan Purpose", ["car", "furniture/equipment", "radio/TV", "domestic appliance", "repairs", "education", "business", "other"])
            applicant_data["installment_commitment"] = st.slider("Installment Rate (% of Income)", min_value=1, max_value=4, value=3)
            applicant_data["existing_credits"] = st.slider("Existing Credits at Bank", min_value=1, max_value=4, value=1)
        with c3:
            st.markdown("**Demographics**")
            applicant_data["age"] = st.number_input("Applicant Age", min_value=18, max_value=85, value=32, step=1)
            applicant_data["employment"] = st.selectbox("Employment Duration", ["<1", "1<=X<4", "4<=X<7", ">=7", "unemployed"], index=1)
            applicant_data["housing"] = st.selectbox("Housing Status", ["own", "rent", "for free"])
            applicant_data["num_dependents"] = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

        with st.expander("Additional Background (Optional Defaults)"):
            ec1, ec2 = st.columns(2)
            with ec1:
                applicant_data["personal_status"] = st.selectbox("Personal Status", ["male single", "female single", "male married/widow", "female div/sep", "male div/sep"])
                applicant_data["other_parties"] = st.selectbox("Co-signers / Guarantors", ["none", "co-applicant", "guarantor"])
                applicant_data["property_magnitude"] = st.selectbox("Collateral / Property", ["real estate", "life insurance", "car", "no property"])
            with ec2:
                applicant_data["other_payment_plans"] = st.selectbox("Other Payment Plans", ["none", "bank", "stores"])
                applicant_data["job"] = st.selectbox("Employment Category", ["skilled", "unskilled resident", "management/self-employed", "unemployed/unskilled non-resident"])
                applicant_data["residence_since"] = st.slider("Years at Present Residence", 1, 4, 2)
                applicant_data["own_telephone"] = st.selectbox("Telephone Registered", ["yes", "none"])
                applicant_data["foreign_worker"] = st.selectbox("Foreign Worker Status", ["yes", "no"])

    submit_btn = st.form_submit_button("⚡ Compute Default Probability & Risk Tier", use_container_width=True)

if submit_btn:
    try:
        res = score_applicant(applicant_data, threshold=threshold)
        prob = res["probability_of_default"]
        label = res["risk_label"]

        st.divider()
        st.subheader("🎯 Risk Assessment Result")
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.metric("Estimated Default Probability", f"{prob * 100:.2f}%")
        with rc2:
            badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[label]
            st.metric("Assigned Risk Tier", f"{badge} {label} Risk")
        with rc3:
            st.metric("Applied Threshold", f"{res['threshold_used']:.2f}")

        st.progress(min(max(prob, 0.0), 1.0))

        if label == "Low":
            st.success("✅ **Recommendation: APPROVE** — Applicant meets credit quality guidelines with minimal default risk.")
        elif label == "Medium":
            st.warning("⚠️ **Recommendation: MANUAL UNDERWRITING** — Borderline profile; require additional income verification.")
        else:
            st.error("❌ **Recommendation: DECLINE / REQUIRE COLLATERAL** — High probability of delinquency detected.")

    except Exception as exc:
        st.error(f"Error scoring applicant: {exc}")

st.divider()
st.markdown(
    "<small><strong>CodeAlpha_CreditScoring</strong> | Built by <strong>Aryan</strong> "
    "(GitHub: <a href='https://github.com/aryannten' target='_blank'>aryannten</a>) | "
    "Task 1: Credit Scoring Model for <em>CodeAlpha Machine Learning Internship</em>. "
    "Educational project, not financial advice.</small>",
    unsafe_allow_html=True,
)
