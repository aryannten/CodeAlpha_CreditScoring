"""Data cleaning, imputation, outlier handling, and pipeline preprocessing.

All preprocessing transformations are encapsulated within scikit-learn
ColumnTransformer pipelines, ensuring transformations are fitted strictly
on training splits and preventing data leakage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPARSE_COLUMNS = ("MonthlyIncome", "NumberOfDependents", "num_dependents")


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean '<col>_was_missing' indicator columns for sparse fields.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with indicator columns added.
    """
    df = df.copy()
    for col in SPARSE_COLUMNS:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def cap_outliers(
    df: pd.DataFrame,
    columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    domain_bounds: Dict[str, Tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Cap extreme outliers using percentile thresholds or explicit domain constraints.

    Winsorizing/capping preserves all records while mitigating the adverse leverage
    of anomalous values on linear models and variance estimators.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature matrix.
    columns : List[str]
        Numeric columns to evaluate.
    lower_quantile : float, default 0.01
        Lower quantile threshold.
    upper_quantile : float, default 0.99
        Upper quantile threshold.
    domain_bounds : Dict[str, Tuple[float, float]], optional
        Manual bounds {column: (min_val, max_val)}.

    Returns
    -------
    pd.DataFrame
        Cleaned feature matrix.
    """
    df = df.copy()
    domain_bounds = domain_bounds or {}

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if col in domain_bounds:
            low_val, high_val = domain_bounds[col]
        else:
            low_val = float(df[col].quantile(lower_quantile))
            high_val = float(df[col].quantile(upper_quantile))

        df[col] = df[col].clip(lower=low_val, upper=high_val)

    return df


def clean_give_me_some_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the 'Give Me Some Credit' dataset schema.

    - Adds missing indicators for MonthlyIncome and NumberOfDependents.
    - Sets biologically impossible age == 0 to NaN for subsequent imputation.
    - Caps revolving utilization to [0.0, 1.0] (unsecured lines cannot have valid >100% usage).
    - Caps heavy-tailed debt ratio, monthly income, and late delinquency counts.

    Parameters
    ----------
    df : pd.DataFrame
        Raw features.

    Returns
    -------
    pd.DataFrame
        Cleaned features.
    """
    df = add_missing_flags(df)

    # Clean impossible age
    for age_col in ("age", "Age"):
        if age_col in df.columns:
            df.loc[df[age_col] <= 0, age_col] = np.nan

    domain_bounds = {
        "RevolvingUtilizationOfUnsecuredLines": (0.0, 1.0),
    }

    numeric_cols = [
        c for c in [
            "RevolvingUtilizationOfUnsecuredLines",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfTimes90DaysLate",
            "NumberOfTimes60-89DaysPastDueNotWorse",
            "NumberOfTimes30-59DaysPastDueNotWorse",
            "NumberOfDependents",
            "age",
        ]
        if c in df.columns
    ]

    return cap_outliers(df, numeric_cols, domain_bounds=domain_bounds)


def clean_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the German Credit dataset schema.

    - Caps extreme credit amounts and durations at the 99th percentile.
    """
    numeric_cols = [
        c for c in ["duration", "credit_amount", "installment_commitment", "age"]
        if c in df.columns
    ]
    return cap_outliers(df, numeric_cols, lower_quantile=0.01, upper_quantile=0.99)


def build_preprocessor(
    X_train: pd.DataFrame,
    categorical_features: List[str] | None = None,
    numeric_features: List[str] | None = None,
) -> ColumnTransformer:
    """Create an unfitted ColumnTransformer for numerical and categorical pipelines.

    - Numerical pipeline: SimpleImputer(strategy='median') -> StandardScaler()
    - Categorical pipeline: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')

    Fitted strictly on training folds to prevent data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features used to infer dtypes if not explicitly supplied.
    categorical_features : List[str], optional
        Explicit categorical column names.
    numeric_features : List[str], optional
        Explicit numeric column names.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer pipeline.
    """
    if categorical_features is None:
        categorical_features = list(
            X_train.select_dtypes(include=["object", "category", "string"]).columns
        )
    if numeric_features is None:
        numeric_features = [
            col for col in X_train.columns if col not in categorical_features
        ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame, target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    X = df.drop(columns=[target_name])
    y = df[target_name]
    return X, y
