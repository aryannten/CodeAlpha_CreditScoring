CELL_SOURCES = [
    r'''
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

RANDOM_STATE = 42
print("Setup complete. random_state =", RANDOM_STATE)
''',
    r'''
from src.data_loader import load_dataset

ds = load_dataset("auto")
print("Dataset:", ds["name"])
print("Source:", ds["source"])
print("Samples:", ds["n_samples"], "| Features:", ds["n_features"])
print("Default rate:", format(ds["default_rate"], ".2%"))
print("Target:", ds["target_name"])
print()
print("Feature names:", ds["feature_names"])
''',
]