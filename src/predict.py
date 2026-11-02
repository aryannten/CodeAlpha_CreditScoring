"""Inference API for real-time applicant credit risk scoring.

Provides:
  - score_applicant(applicant_dict, threshold): Returns probability of default,
    risk band ('Low' / 'Medium' / 'High'), and the decision threshold used.

Handles raw applicant dictionaries, automatically executes domain feature engineering,
and feeds into the serialized scikit-learn / imblearn pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
import pandas as pd

from .features import engineer_features

MODEL_PATH = os.path.join("models", "best_model.joblib")
DEFAULT_DECISION_THRESHOLD = 0.40


def _load_model(model_path: str = MODEL_PATH) -> Any:
    """Load serialized pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved credit scoring model not found at '{model_path}'. "
            "Execute training first via: python -m src.train"
        )
    return joblib.load(model_path)


def _get_expected_features(pipeline: Any) -> List[str]:
    """Retrieve feature column names expected by the preprocessor."""
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)
    return []


def classify_risk(probability: float, threshold: float = DEFAULT_DECISION_THRESHOLD) -> str:
    """Map probability of default into retail credit risk tiers.

    Risk bands:
      - Low: Default probability < 0.20 (Creditworthy, streamline approval)
      - Medium: 0.20 <= Default probability < threshold (Manual underwriting review required)
      - High: Default probability >= threshold (Adverse credit risk, decline or require collateral)
    """
    if probability < 0.20:
        return "Low"
    elif probability < threshold:
        return "Medium"
    else:
        return "High"


def score_applicant(
    applicant_dict: Dict[str, Any],
    model_path: str = MODEL_PATH,
    threshold: float = DEFAULT_DECISION_THRESHOLD,
) -> Dict[str, Any]:
    """Score a loan applicant and compute risk metrics.

    Parameters
    ----------
    applicant_dict : Dict[str, Any]
        Raw applicant features.
    model_path : str, default 'models/best_model.joblib'
        Path to serialized model artifact.
    threshold : float, default 0.40
        Decision threshold for high-risk categorization.

    Returns
    -------
    Dict[str, Any]
        Contains 'probability_of_default', 'risk_label', and 'threshold_used'.
    """
    pipeline = _load_model(model_path)

    # 1. Convert applicant to single-row DataFrame
    df_raw = pd.DataFrame([applicant_dict])

    # 2. Apply feature engineering
    df_eng, _ = engineer_features(df_raw)

    # 3. Match expected input columns for pipeline
    expected_cols = _get_expected_features(pipeline)
    if expected_cols:
        # Reindex ensures all expected columns exist (missing are NaN, filled by imputer)
        df_input = df_eng.reindex(columns=expected_cols)
    else:
        df_input = df_eng

    # 4. Predict probability of default (Class 1)
    if hasattr(pipeline, "predict_proba"):
        prob_default = float(pipeline.predict_proba(df_input)[:, 1][0])
    else:
        decision = pipeline.decision_function(df_input)
        prob_default = float(1.0 / (1.0 + np.exp(-decision[0])))

    risk_label = classify_risk(prob_default, threshold=threshold)

    return {
        "probability_of_default": round(prob_default, 4),
        "risk_label": risk_label,
        "threshold_used": round(threshold, 2),
    }


if __name__ == "__main__":
    # Test sample with German Credit schema
    sample_applicant = {
        "checking_status": "<0",
        "duration": 24,
        "credit_history": "critical",
        "purpose": "car",
        "credit_amount": 3500,
        "savings_status": "<100",
        "employment": "1<=X<4",
        "installment_commitment": 4,
        "personal_status": "male single",
        "other_parties": "none",
        "residence_since": 2,
        "property_magnitude": "car",
        "age": 28,
        "other_payment_plans": "none",
        "housing": "rent",
        "existing_credits": 1,
        "job": "skilled",
        "num_dependents": 1,
        "own_telephone": "none",
        "foreign_worker": "yes",
    }
    try:
        res = score_applicant(sample_applicant)
        print("Applicant risk score:", res)
    except FileNotFoundError as err:
        print(err)
