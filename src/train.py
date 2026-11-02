"""Model training, hyperparameter optimization, and imbalance handling.

Trains and tunes four classification algorithms:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost (Bonus model)

Imbalance Strategies compared for each model:
  (a) Cost-sensitive learning: class_weight='balanced' (or scale_pos_weight for XGBoost)
  (b) Resampling: SMOTE applied INSIDE each training cross-validation fold via imblearn Pipeline.

All tuning uses 5-Fold Stratified Cross-Validation scored on ROC-AUC (random_state=42).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAS_IMB = True
except ImportError:
    HAS_IMB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from .data_loader import split_train_test
from .features import engineer_features
from .preprocessing import (
    build_preprocessor,
    clean_german_credit,
    clean_give_me_some_credit,
)

RANDOM_STATE = 42
N_CV = 5
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join("reports", "figures")
MODELS_DIR = "models"


def _ensure_dirs() -> None:
    for d in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)


def get_model_specifications(pos_weight: float = 1.0) -> List[Dict[str, Any]]:
    """Return model architectures and search spaces for hyperparameter tuning."""
    specs = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            ),
            "param_dist": {
                "C": np.logspace(-3, 2, 10),
                "solver": ["lbfgs", "liblinear"],
            },
            "supports_class_weight": True,
        },
        {
            "name": "Decision Tree",
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),
            "param_dist": {
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
            },
            "supports_class_weight": True,
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_dist": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 16, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            "supports_class_weight": True,
        },
    ]

    if HAS_XGB:
        specs.append(
            {
                "name": "XGBoost",
                "estimator": XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                    eval_metric="logloss",
                ),
                "param_dist": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.85, 1.0],
                    "colsample_bytree": [0.7, 0.85, 1.0],
                },
                "supports_class_weight": False,
                "pos_weight": pos_weight,
            }
        )

    return specs


def train_and_tune_model(
    model_name: str,
    base_model: Any,
    param_dist: Dict[str, Any],
    preprocessor: Any,
    use_smote: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Build and fit a pipeline via RandomizedSearchCV without data leakage."""
    steps = [("preprocessor", preprocessor)]

    if use_smote:
        if not HAS_IMB:
            raise RuntimeError("imblearn is required for SMOTE.")
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        pipeline_cls = ImbPipeline
    else:
        pipeline_cls = SkPipeline

    steps.append(("classifier", base_model))
    pipe = pipeline_cls(steps=steps)

    prefixed_params = {f"classifier__{k}": v for k, v in param_dist.items()}

    cv = StratifiedKFold(n_splits=N_CV, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=prefixed_params,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    cv_roc_auc = float(search.best_score_)

    if hasattr(best_estimator, "predict_proba"):
        y_proba = best_estimator.predict_proba(X_test)[:, 1]
    else:
        scores = best_estimator.decision_function(X_test)
        y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    test_roc_auc = float(roc_auc_score(y_test, y_proba))

    return {
        "model": model_name,
        "imbalance": "smote" if use_smote else "class_weight_balanced",
        "cv_roc_auc": cv_roc_auc,
        "test_roc_auc": test_roc_auc,
        "best_params": search.best_params_,
        "use_smote": use_smote,
        "estimator": best_estimator,
        "y_proba_test": y_proba,
    }


def compare_feature_engineering_lift(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    X_train_eng: pd.DataFrame,
    X_test_eng: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate performance before vs after feature engineering."""
    rows = []
    models_to_test = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ]

    for name, clf in models_to_test:
        # Before feature engineering
        prep_raw = build_preprocessor(X_train_raw)
        pipe_raw = SkPipeline([("preprocessor", prep_raw), ("clf", clf)])
        pipe_raw.fit(X_train_raw, y_train)
        proba_raw = pipe_raw.predict_proba(X_test_raw)[:, 1]
        auc_raw = roc_auc_score(y_test, proba_raw)

        # After feature engineering
        prep_eng = build_preprocessor(X_train_eng)
        pipe_eng = SkPipeline([("preprocessor", prep_eng), ("clf", clf)])
        pipe_eng.fit(X_train_eng, y_train)
        proba_eng = pipe_eng.predict_proba(X_test_eng)[:, 1]
        auc_eng = roc_auc_score(y_test, proba_eng)

        rows.append({
            "model": name,
            "raw_features_roc_auc": round(float(auc_raw), 4),
            "engineered_features_roc_auc": round(float(auc_eng), 4),
            "lift": round(float(auc_eng - auc_raw), 4),
        })

    lift_df = pd.DataFrame(rows)
    print("\n=== Feature Engineering Lift (Before vs. After) ===")
    print(lift_df.to_string(index=False))
    lift_df.to_csv(os.path.join(REPORTS_DIR, "feature_engineering_lift.csv"), index=False)
    return lift_df


def train_all(ds: Dict[str, Any]) -> Dict[str, Any]:
    """Execute end-to-end training and comparative analysis."""
    _ensure_dirs()
    X = ds["X"].copy()
    y = ds["y"].copy()
    dataset_name = ds["name"]

    # 1. Clean features
    if "Give Me Some Credit" in dataset_name:
        X_clean = clean_give_me_some_credit(X)
    else:
        X_clean = clean_german_credit(X)

    # 2. Split raw data for baseline comparison
    X_tr_raw, X_te_raw, y_train, y_test = split_train_test(
        X_clean, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3. Engineer features
    X_eng, engineered_cols = engineer_features(X_clean, dataset_name)
    X_train, X_test, _, _ = split_train_test(
        X_eng, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"\n==========================================")
    print(f"Training Pipeline on: {dataset_name}")
    print(f"Total Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Target Default Rate: {y.mean():.2%}")
    print(f"Engineered Features Added: {len(engineered_cols)} -> {list(engineered_cols)}")
    print(f"==========================================\n")

    # Evaluate feature engineering impact
    lift_df = compare_feature_engineering_lift(
        X_tr_raw, X_te_raw, X_train, X_test, y_train, y_test
    )

    # Build preprocessor on training data
    preprocessor = build_preprocessor(X_train)

    # Class imbalance ratio for cost-sensitive models
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    pos_weight = float(neg_count / max(pos_count, 1))

    specs = get_model_specifications(pos_weight=pos_weight)
    all_results = []
    best_estimator = None
    best_auc = -1.0
    best_info: Dict[str, Any] = {}

    for spec in specs:
        model_name = spec["name"]
        param_dist = spec["param_dist"]

        # Strategy 1: class_weight='balanced' or scale_pos_weight
        if spec.get("supports_class_weight", False):
            clf = spec["estimator"].set_params(class_weight="balanced")
            res_bal = train_and_tune_model(
                model_name=model_name,
                base_model=clf,
                param_dist=param_dist,
                preprocessor=preprocessor,
                use_smote=False,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            all_results.append(res_bal)
            print(f"  [{model_name} | class_weight='balanced'] CV ROC-AUC: {res_bal['cv_roc_auc']:.4f} | Test ROC-AUC: {res_bal['test_roc_auc']:.4f}")
            if res_bal["test_roc_auc"] > best_auc:
                best_auc = res_bal["test_roc_auc"]
                best_estimator = res_bal["estimator"]
                best_info = res_bal
        elif model_name == "XGBoost":
            clf = spec["estimator"].set_params(scale_pos_weight=spec["pos_weight"])
            res_bal = train_and_tune_model(
                model_name=model_name,
                base_model=clf,
                param_dist=param_dist,
                preprocessor=preprocessor,
                use_smote=False,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            res_bal["imbalance"] = "scale_pos_weight"
            all_results.append(res_bal)
            print(f"  [{model_name} | scale_pos_weight] CV ROC-AUC: {res_bal['cv_roc_auc']:.4f} | Test ROC-AUC: {res_bal['test_roc_auc']:.4f}")
            if res_bal["test_roc_auc"] > best_auc:
                best_auc = res_bal["test_roc_auc"]
                best_estimator = res_bal["estimator"]
                best_info = res_bal

        # Strategy 2: SMOTE inside training pipeline
        if HAS_IMB:
            clf_raw = spec["estimator"]
            res_smote = train_and_tune_model(
                model_name=model_name,
                base_model=clf_raw,
                param_dist=param_dist,
                preprocessor=preprocessor,
                use_smote=True,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            all_results.append(res_smote)
            print(f"  [{model_name} | SMOTE] CV ROC-AUC: {res_smote['cv_roc_auc']:.4f} | Test ROC-AUC: {res_smote['test_roc_auc']:.4f}")
            if res_smote["test_roc_auc"] > best_auc:
                best_auc = res_smote["test_roc_auc"]
                best_estimator = res_smote["estimator"]
                best_info = res_smote

    # Save best model to disk
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_estimator, best_path)
    print(f"\n>>> Best overall model saved to: {best_path}")
    print(f"    Selected: {best_info['model']} with {best_info['imbalance']} (Test ROC-AUC: {best_auc:.4f})")

    summary = {
        "dataset": dataset_name,
        "n_samples": len(X),
        "default_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "n_cv": N_CV,
        "results": all_results,
        "best_model": {
            "model": best_info["model"],
            "imbalance": best_info["imbalance"],
            "test_roc_auc": best_info["test_roc_auc"],
            "cv_roc_auc": best_info["cv_roc_auc"],
            "best_params": best_info["best_params"],
        },
        "feature_engineering_lift": lift_df.to_dict(orient="records"),
        "engineered_cols": engineered_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    # Save JSON summary (without non-serializable objects)
    json_summary = {
        "dataset": dataset_name,
        "default_rate": float(y.mean()),
        "best_model": summary["best_model"],
        "results": [
            {
                "model": r["model"],
                "imbalance": r["imbalance"],
                "cv_roc_auc": r["cv_roc_auc"],
                "test_roc_auc": r["test_roc_auc"],
                "best_params": r["best_params"],
            }
            for r in all_results
        ],
        "feature_engineering_lift": summary["feature_engineering_lift"],
    }
    with open(os.path.join(REPORTS_DIR, "training_results.json"), "w") as f:
        json.dump(json_summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    from .data_loader import load_dataset
    dataset = load_dataset("auto")
    train_all(dataset)
