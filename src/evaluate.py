"""Evaluation suite: metrics calculation, comparative visualizations, and interpretability.

Computes Precision, Recall, F1-score, ROC-AUC, Accuracy, and PR-AUC.
Generates:
  - Confusion matrix heatmaps for each model
  - Multi-model ROC curves
  - Multi-model Precision-Recall curves
  - Decision threshold sensitivity plot (Recall vs Precision trade-off)
  - Top-10 feature importance / Logistic Regression coefficient bars
  - SHAP summary plot for model explainability
  - models x metrics CSV comparison table
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join("reports", "figures")
sns.set_theme(style="whitegrid")


def _ensure_dirs() -> None:
    for d in (REPORTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics at a defined decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "threshold": float(threshold),
    }


def explain_accuracy_paradox(default_rate: float) -> str:
    """Return a plain-English explanation of why accuracy is deceptive on imbalanced credit data."""
    majority_pct = (1.0 - default_rate) * 100
    default_pct = default_rate * 100
    return (
        f"Why Accuracy is Misleading on Imbalanced Data:\n"
        f"In credit scoring, the minority default class represents only {default_pct:.1f}% of applicants, "
        f"while {majority_pct:.1f}% are non-defaulters.\n"
        f"A naive 'dummy' classifier that approves every applicant and predicts 0% defaults would achieve "
        f"{majority_pct:.1f}% accuracy, yet fail to catch a single defaulter (Recall = 0.0)!\n"
        f"In retail lending, approving a borrower who defaults can cost thousands of dollars (loss of principal), "
        f"whereas rejecting a creditworthy applicant only foregoes interest margin.\n"
        f"Hence, ROC-AUC, PR-AUC, and Recall on the default class are far more critical evaluation metrics."
    )


def plot_confusion_matrix(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Save an annotated confusion matrix heatmap."""
    _ensure_dirs()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curves(
    results: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: str,
) -> None:
    """Plot multi-model ROC curves on a single figure."""
    _ensure_dirs()
    plt.figure(figsize=(8, 6.5))
    for name, y_true, y_proba in results:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curves(
    results: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: str,
) -> None:
    """Plot multi-model Precision-Recall curves on a single figure."""
    _ensure_dirs()
    plt.figure(figsize=(8, 6.5))
    for name, y_true, y_proba in results:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Detection of Defaulters)", fontsize=11)
    plt.ylabel("Precision (Accuracy of Default Flags)", fontsize=11)
    plt.title("Precision-Recall (PR) Curves — Imbalanced Credit Risk", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_threshold_tuning(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    save_path: str,
    chosen_threshold: float = 0.40,
) -> None:
    """Plot Precision, Recall, and F1 across decision thresholds."""
    _ensure_dirs()
    thresholds = np.linspace(0.05, 0.90, 80)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(8, 5.5))
    plt.plot(thresholds, precisions, label="Precision", color="#1f77b4", lw=2)
    plt.plot(thresholds, recalls, label="Recall (Defaulters Caught)", color="#d62728", lw=2)
    plt.plot(thresholds, f1s, label="F1-Score", color="#2ca02c", lw=2, linestyle=":")
    plt.axvline(
        chosen_threshold,
        color="black",
        linestyle="--",
        lw=1.8,
        label=f"Selected Threshold = {chosen_threshold:.2f}",
    )
    plt.title(f"Threshold Tuning & Lending Trade-off — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Decision Probability Threshold", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.legend(loc="center right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_feature_importance(
    estimator: Any,
    input_feature_names: List[str],
    model_name: str,
    save_path: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Extract and plot top-N feature importances or logistic regression coefficients."""
    _ensure_dirs()

    # Unwrap pipeline to reach classifier and preprocessor
    clf = estimator
    preprocessor = None
    if hasattr(estimator, "named_steps"):
        preprocessor = estimator.named_steps.get("preprocessor")
        clf = estimator.named_steps.get("classifier") or estimator.named_steps.get("clf")

    importances = None
    metric_label = "Feature Importance"

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        metric_label = "Gini / Split Importance"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_.ravel())
        metric_label = "Absolute Logistic Regression Coefficient (|beta|)"

    if importances is None:
        return pd.DataFrame()

    # Get transformed column names from ColumnTransformer
    transformed_names = input_feature_names
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            transformed_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pass

    if len(transformed_names) != len(importances):
        # Fallback names
        transformed_names = [f"Feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "feature": transformed_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(8.5, 5))
    palette = sns.color_palette("mako", len(df_imp))
    sns.barplot(
        data=df_imp.iloc[::-1],
        x="importance",
        y="feature",
        palette=palette,
    )
    plt.title(f"Top {top_n} Risk Drivers — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel(metric_label, fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return df_imp


def plot_shap_summary(
    estimator: Any,
    X_sample: pd.DataFrame,
    save_path: str,
    max_display: int = 10,
) -> None:
    """Generate and save SHAP summary beeswarm plot for the best model."""
    if not HAS_SHAP:
        print("[evaluate] SHAP not available; skipping SHAP plot.")
        return

    _ensure_dirs()
    try:
        preprocessor = estimator.named_steps.get("preprocessor")
        clf = estimator.named_steps.get("classifier") or estimator.named_steps.get("clf")

        if preprocessor is None or clf is None:
            return

        X_trans = preprocessor.transform(X_sample)
        feature_names = (
            list(preprocessor.get_feature_names_out())
            if hasattr(preprocessor, "get_feature_names_out")
            else None
        )

        plt.figure(figsize=(9, 6))
        # Use TreeExplainer for tree models, LinearExplainer for linear, or Explainer
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans)
            # If binary classification returns list [shap_class_0, shap_class_1]
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            shap.summary_plot(
                shap_values,
                X_trans,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )
        else:
            explainer = shap.Explainer(clf, X_trans)
            shap_values = explainer(X_trans)
            shap.summary_plot(
                shap_values,
                features=X_trans,
                feature_names=feature_names,
                max_display=max_display,
                show=False,
            )

        plt.title("SHAP Summary Plot — Feature Impact on Default Risk", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[evaluate] SHAP summary plot saved to: {save_path}")
    except Exception as exc:
        print(f"[evaluate] Could not generate SHAP plot: {exc}")


def save_comparison_table(rows: List[Dict[str, Any]], save_path: str) -> pd.DataFrame:
    """Persist final model comparison table as CSV and return DataFrame."""
    _ensure_dirs()
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"\nSaved models comparison table to: {save_path}")
    print(df.to_string(index=False))
    return df
