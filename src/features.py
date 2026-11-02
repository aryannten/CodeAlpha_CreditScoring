"""Feature engineering for credit risk scoring.

Each feature is grounded in credit risk and retail banking domain principles:
A 3rd-year engineering student can understand why each feature predicts risk.

1. Give Me Some Credit features:
   - total_past_due_count: Aggregates short-term, medium-term, and severe delinquencies.
     A single metric for overall payment failure history.
   - has_any_late_payment: Binary indicator; even a single missed payment is an adverse signal.
   - income_per_dependent: MonthlyIncome / (NumberOfDependents + 1). Measures remaining
     financial buffer per household member.
   - est_monthly_debt_payment: MonthlyIncome * DebtRatio. Absolute monthly debt obligation.
   - age_bucket: Life stage categorization (Young <30, Adult 30-60, Senior >60).
   - utilization_bucket: Revolving line utilization bands (Low <0.3, Medium 0.3-0.7, High >0.7).
   - log_*: Log1p transformations on heavy-tailed financial variables (income, debt ratio).

2. German Credit features:
   - monthly_installment_burden: credit_amount / duration. Immediate monthly cash outflow.
   - credit_to_age_ratio: credit_amount / age. Loan leverage relative to life stage.
   - has_critical_credit: Binary indicator for critical / delayed past credit history.
   - has_no_checking: Binary indicator for lack of established checking account relationship.
   - age_bucket: Life stage segmentation (Young <30, Adult 30-60, Senior >60).
   - log_credit_amount, log_duration: Compresses positive skew for linear and distance models.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def engineer_features(
    df: pd.DataFrame, dataset_name: str | None = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate domain-specific credit risk features for tabular credit data.

    Adapts dynamically to the dataset schema (Give Me Some Credit or German Credit)
    without hardcoding column prerequisites or failing on missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input cleaned feature matrix.
    dataset_name : str, optional
        Dataset indicator to select specialized logic.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        A new DataFrame with engineered features appended, and a list of new feature names.
    """
    df = df.copy()
    engineered: List[str] = []

    # Map candidate column names
    col_map = {}
    candidates = {
        "revolving_util": ["RevolvingUtilizationOfUnsecuredLines"],
        "debt_ratio": ["DebtRatio"],
        "monthly_income": ["MonthlyIncome"],
        "num_dependents": ["NumberOfDependents", "num_dependents"],
        "age": ["age", "Age"],
        "late_30_59": [
            "NumberOfTimes30-59DaysPastDueNotWorse",
            "NumberOfTime30-59DaysPastDueNotWorse",
        ],
        "late_60_89": [
            "NumberOfTimes60-89DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
        ],
        "late_90": ["NumberOfTimes90DaysLate", "NumberOfTime90DaysLate"],
        "duration": ["duration", "Duration"],
        "credit_amount": ["credit_amount", "CreditAmount"],
        "installment_commitment": ["installment_commitment"],
        "credit_history": ["credit_history"],
        "checking_status": ["checking_status"],
    }

    for target_key, candidate_list in candidates.items():
        for cand in candidate_list:
            if cand in df.columns:
                col_map[target_key] = cand
                break

    def c(key: str) -> str | None:
        return col_map.get(key)

    # =========================================================================
    # A. GIVE ME SOME CREDIT SCHEMA FEATURES
    # =========================================================================

    # 1. Total past-due count
    late_cols = [c(k) for k in ("late_30_59", "late_60_89", "late_90") if c(k)]
    if late_cols:
        df["total_past_due_count"] = df[late_cols].sum(axis=1)
        engineered.append("total_past_due_count")

    # 2. Has any late payment flag
    if "total_past_due_count" in df.columns:
        df["has_any_late_payment"] = (df["total_past_due_count"] > 0).astype(int)
        engineered.append("has_any_late_payment")

    # 3. Income per dependent / family member
    inc_col = c("monthly_income")
    dep_col = c("num_dependents")
    if inc_col and dep_col:
        df["income_per_dependent"] = df[inc_col] / (df[dep_col].fillna(0) + 1.0)
        engineered.append("income_per_dependent")

    # 4. Estimated monthly debt payment
    ratio_col = c("debt_ratio")
    if inc_col and ratio_col:
        df["est_monthly_debt_payment"] = df[inc_col] * df[ratio_col]
        engineered.append("est_monthly_debt_payment")

    # 5. Revolving credit utilization bucket
    util_col = c("revolving_util")
    if util_col:
        df["utilization_bucket"] = pd.cut(
            df[util_col],
            bins=[-np.inf, 0.3, 0.7, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)
        engineered.append("utilization_bucket")

    # 6. Log transforms for skewed financial features
    for src_key, out_col in [
        ("monthly_income", "log_monthly_income"),
        ("debt_ratio", "log_debt_ratio"),
        ("est_monthly_debt_payment", "log_est_monthly_debt_payment"),
    ]:
        source = c(src_key) if src_key in col_map else (src_key if src_key in df.columns else None)
        if source and source in df.columns:
            clipped = df[source].clip(lower=0)
            df[out_col] = np.log1p(clipped)
            engineered.append(out_col)

    # =========================================================================
    # B. GERMAN CREDIT SCHEMA FEATURES
    # =========================================================================

    # 7. Monthly installment burden (Credit Amount / Duration in months)
    dur_col = c("duration")
    amt_col = c("credit_amount")
    if amt_col and dur_col:
        df["monthly_installment_burden"] = df[amt_col] / (df[dur_col].clip(lower=1))
        engineered.append("monthly_installment_burden")

    # 8. Credit exposure relative to age
    age_col = c("age")
    if amt_col and age_col:
        df["credit_to_age_ratio"] = df[amt_col] / (df[age_col].clip(lower=18))
        engineered.append("credit_to_age_ratio")

    # 9. Adverse past credit history indicator
    hist_col = c("credit_history")
    if hist_col:
        df["has_critical_credit"] = (
            df[hist_col].astype(str).str.lower().isin(["critical", "delayed"])
        ).astype(int)
        engineered.append("has_critical_credit")

    # 10. No checking account indicator
    chk_col = c("checking_status")
    if chk_col:
        df["has_no_checking"] = (
            df[chk_col].astype(str).str.lower().str.contains("no checking")
        ).astype(int)
        engineered.append("has_no_checking")

    # 11. Log transforms for German Credit continuous variables
    if amt_col and "log_credit_amount" not in df.columns:
        df["log_credit_amount"] = np.log1p(df[amt_col].clip(lower=0))
        engineered.append("log_credit_amount")

    if dur_col and "log_duration" not in df.columns:
        df["log_duration"] = np.log1p(df[dur_col].clip(lower=0))
        engineered.append("log_duration")

    # =========================================================================
    # C. GENERAL FEATURES (Both schemas)
    # =========================================================================

    # 12. Age life stage buckets: Young (<30), Adult (30-60), Senior (>60)
    if age_col and "age_bucket" not in df.columns:
        df["age_bucket"] = pd.cut(
            df[age_col],
            bins=[-np.inf, 30, 60, np.inf],
            labels=["Young", "Adult", "Senior"],
        ).astype(str)
        engineered.append("age_bucket")

    return df, engineered


def describe_feature_logic() -> List[str]:
    """Plain-English business rationale for all engineered features."""
    return [
        "total_past_due_count: Aggregates 30-59, 60-89, and 90+ days delinquencies. "
        "Borrowers with frequent past-due occurrences are statistically much more likely to default.",
        "has_any_late_payment: Binary flag indicating any prior delinquency. In retail lending, "
        "any delinquency is an adverse credit bureau flag.",
        "income_per_dependent: MonthlyIncome / (NumberOfDependents + 1). Normalizes income by family size, "
        "measuring disposable financial headroom.",
        "est_monthly_debt_payment: MonthlyIncome * DebtRatio. Absolute monthly debt obligation, "
        "identifying borrowers with heavy fixed monthly debt payments.",
        "monthly_installment_burden: credit_amount / duration. Estimates the immediate monthly principal "
        "commitment required from the borrower.",
        "credit_to_age_ratio: credit_amount / age. Measures leverage relative to earning lifecycle; "
        "young borrowers taking large loans carry higher repayment volatility.",
        "has_critical_credit: 1 if past credit history was marked critical or delayed, identifying "
        "repeat delinquency risks.",
        "has_no_checking: 1 if the applicant has no active checking account relationship with the bank.",
        "age_bucket: Young (<30), Adult (30-60), Senior (>60). Life-stage categorization captures "
        "structural career stability differences.",
        "utilization_bucket: Low (<0.3), Medium (0.3-0.7), High (>0.7). Credit capacity utilization "
        "is among the strongest credit score drivers.",
        "log_*: Log1p transform tames right-skewed distributions (income, credit amounts, debt), "
        "preventing extreme outliers from distorting linear and distance-based estimators.",
    ]


if __name__ == "__main__":
    print("Feature engineering module loaded successfully.")
    for line in describe_feature_logic():
        print(" -", line)
