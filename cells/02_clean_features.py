CELL_SOURCES = [
    r'''
from src.preprocessing import clean_give_me_some_credit

X_clean = clean_give_me_some_credit(X)
print("Cleaned shape:", X_clean.shape)
miss = X_clean.isna().sum()
print("Missing after clean:")
print(miss[miss > 0])
''',
    r'''
from src.features import engineer_features, describe_feature_logic

for line in describe_feature_logic():
    print("- ", line)

X_eng, engineered_cols = engineer_features(X_clean, ds["name"])
print()
print("Engineered features:", engineered_cols)
''',
]