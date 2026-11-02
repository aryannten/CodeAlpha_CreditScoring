CELL_SOURCES = [
    r'''
from src.evaluate import (
    save_comparison_table, compute_metrics, plot_confusion_matrix,
)

fig_dir = "reports/figures"
comparison_rows = []
roc_results = []  # (display_name, y_true, y_proba)

for r in summary["results"]:
    name = r["model"] + " (" + r["imbalance"] + ")"
    y_proba = r["y_proba_test"]
    y_pred = (y_proba >= 0.5).astype(int)

    m = compute_metrics(y_test.values, y_proba, threshold=0.5)
    comparison_rows.append({
        "model": r["model"],
        "imbalance": r["imbalance"],
        "accuracy": round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
        "roc_auc": round(m["roc_auc"], 4),
        "pr_auc": round(m["pr_auc"], 4),
        "cv_roc_auc": round(r["cv_roc_auc"], 4),
        "test_roc_auc": round(r["test_roc_auc"], 4),
    })
    roc_results.append((name, y_test.values, y_proba))

    plot_confusion_matrix(
        name, y_test.values, y_pred,
        os.path.join(fig_dir, "confusion_matrix_" +
                     r["model"].replace(" ", "_") + "_" +
                     r["imbalance"] + ".png"),
    )

save_comparison_table(comparison_rows, "reports/comparison.csv")
print("Comparison table saved to reports/comparison.csv")
print()
print(comparison_rows)
''',
    r'''
from src.evaluate import plot_roc_curves, plot_pr_curve

plot_roc_curves(roc_results, os.path.join(fig_dir, "roc_curves.png"))
plot_pr_curve(roc_results, os.path.join(fig_dir, "pr_curve.png"))
print("ROC curves and PR curve saved to reports/figures/")
''',
    r'''
import joblib
from src.evaluate import plot_threshold_tuning

pipe = joblib.load("models/best_model.joblib")
y_proba = pipe.predict_proba(X_test)[:, 1]
best = summary["best_model"]
print("Best model:", best)

plot_threshold_tuning(
    y_test.values, y_proba,
    best["model"] + " " + best["imbalance"],
    os.path.join(fig_dir, "threshold_tuning.png"),
    chosen_threshold=0.5,
)
print("Threshold tuning plot saved.")

for t in (0.3, 0.5, 0.7):
    m = compute_metrics(y_test.values, y_proba, threshold=t)
    print("threshold=", t, "precision=", format(m["precision"], ".3f"),
          "recall=", format(m["recall"], ".3f"),
          "f1=", format(m["f1"], ".3f"))
''',
    r'''
from src.evaluate import plot_feature_importance

# Locate the fitted estimator for the best model variant.
best_row = None
for cand in summary["results"]:
    if (cand["model"] == summary["best_model"]["model"] and
            cand["imbalance"] == summary["best_model"]["imbalance"]):
        best_row = cand
        break

plot_feature_importance(
    best_row["estimator"],
    list(X_test.columns),
    best["model"] + " " + best["imbalance"],
    os.path.join(fig_dir, "feature_importance.png"),
    top_n=10,
)
print("Feature importance plot saved to reports/figures/")
''',
]