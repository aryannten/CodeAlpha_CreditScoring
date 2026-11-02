CELL_SOURCES = [
    r'''
from src.data_loader import split_train_test

X_train, X_test, y_train, y_test = split_train_test(
    X_eng, y, test_size=0.2, random_state=RANDOM_STATE
)
print("Train:", X_train.shape[0], "| Test:", X_test.shape[0])
print("Train default rate:", format(y_train.mean(), ".4f"))
print("Test default rate:", format(y_test.mean(), ".4f"))
''',
    r'''
from src.train import train_all

summary = train_all(ds, use_engineered=True)

print()
print("=== RESULTS TABLE ===")
for r in summary["results"]:
    print(r["model"], "|", r["imbalance"], "|",
          "CV=", format(r["cv_roc_auc"], ".4f"), "|",
          "Test=", format(r["test_roc_auc"], ".4f"))

print()
print("Best:", summary["best_model"])
''',
]