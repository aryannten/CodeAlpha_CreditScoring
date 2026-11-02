"""Dataset loaders for the credit-scoring project.

Two datasets are supported with a unified ``load_dataset(name)`` entry point:

1. PRIMARY — "Give Me Some Credit" (Kaggle cs-training.csv):
   Predicts whether a borrower will experience 90+ days delinquency in the next
   2 years. Target: SeriousDlqin2yrs (1 = default/delinquent, 0 = non-default).
   150,000 samples, 10 continuous financial features, ~6.68% default rate.

2. SECONDARY — German Credit Data (UCI / OpenML "credit-g", version 1):
   Predicts loan creditworthiness. Target: 1 = bad credit (default analogue),
   0 = good credit. 1,000 samples, 20 financial & demographic features, 30.0% default rate.

Every loader returns a standardized dictionary:
  - X: pd.DataFrame (features)
  - y: pd.Series (binary target: 1 = default / bad credit, 0 = good / non-default)
  - name: str
  - target_name: str
  - feature_names: list of str
  - default_rate: float
  - source: str
  - n_samples: int
  - n_features: int
"""

from __future__ import annotations

import os
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

GMSC_LOCAL_PATH = os.path.join("data", "cs-training.csv")
GERMAN_LOCAL_PATH = os.path.join("data", "german_credit.csv")

# Public mirrors for cs-training.csv (Give Me Some Credit)
GMSC_MIRROR_URLS = [
    "https://raw.githubusercontent.com/mesfind/datasets/master/cs-training.csv",
    "https://raw.githubusercontent.com/rcalix1/Deep-learning-ML-and-tensorflow/master/cs-training.csv",
]


def _read_csv_flexible(path_or_buffer) -> pd.DataFrame:
    """Read CSV with automatic delimiter detection (comma vs semicolon)."""
    try:
        df = pd.read_csv(path_or_buffer)
        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
                df = pd.read_csv(path_or_buffer, sep=";")
            elif isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
                df = pd.read_csv(path_or_buffer, sep=";")
        return df
    except Exception:
        # Fallback to engine='python'
        if isinstance(path_or_buffer, str) and os.path.exists(path_or_buffer):
            return pd.read_csv(path_or_buffer, sep=None, engine="python")
        raise


def load_give_me_some_credit(filepath: str = GMSC_LOCAL_PATH) -> Dict:
    """Load the 'Give Me Some Credit' (cs-training.csv) dataset.

    Checks local data directory first. If not found, attempts public mirrors.
    Raises ValueError with instructions if unavailable, enabling graceful fallback.

    Parameters
    ----------
    filepath : str
        Local path where cs-training.csv is expected.

    Returns
    -------
    dict
        Dataset container with X, y, metadata.
    """
    df = None
    source_used = None

    if os.path.exists(filepath):
        try:
            df = _read_csv_flexible(filepath)
            source_used = f"local file: {filepath}"
        except Exception as exc:
            print(f"[data_loader] Failed reading {filepath}: {exc}")

    if df is None:
        import requests
        for url in GMSC_MIRROR_URLS:
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    df = _read_csv_flexible(StringIO(resp.text))
                    source_used = url
                    os.makedirs(os.path.dirname(filepath) or "data", exist_ok=True)
                    df.to_csv(filepath, index=False)
                    break
            except Exception:
                continue

    if df is None:
        raise ValueError(
            "Could not load 'Give Me Some Credit' (cs-training.csv).\n"
            "To use this dataset:\n"
            "  1. Download 'cs-training.csv' from Kaggle: "
            "https://www.kaggle.com/c/GiveMeSomeCredit/data\n"
            f"  2. Place it at: '{filepath}'\n"
            "Falling back gracefully to the German Credit dataset."
        )

    return _package_give_me_some_credit(df, source=source_used)


def _package_give_me_some_credit(df: pd.DataFrame, source: str) -> Dict:
    """Standardize Give Me Some Credit dataframe into standard format."""
    # Clean up column names and drop index column
    df.columns = [str(c).strip().strip(";") for c in df.columns]
    id_cols = [c for c in df.columns if c.lower() in ("unnamed: 0", "id", "")]
    if id_cols:
        df = df.drop(columns=id_cols)

    target_col = None
    for cand in ("SeriousDlqin2yrs", "seriousdlqin2yrs", "default", "Default"):
        matches = [c for c in df.columns if c.lower() == cand.lower()]
        if matches:
            target_col = matches[0]
            break

    if target_col is None:
        raise ValueError(f"Target column not found in Give Me Some Credit. Found: {list(df.columns)}")

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    return {
        "X": X,
        "y": y,
        "name": "Give Me Some Credit (cs-training.csv)",
        "target_name": target_col,
        "feature_names": list(X.columns),
        "default_rate": float(y.mean()),
        "source": source,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
    }


def load_german_credit(filepath: str = GERMAN_LOCAL_PATH) -> Dict:
    """Load the German Credit dataset (UCI / OpenML 'credit-g', version 1).

    Checks local cache first; if not present, fetches from OpenML and caches
    locally for fast offline use.

    Target Mapping:
      - 1 = 'bad' credit (Defaulter / High Risk - the positive class)
      - 0 = 'good' credit (Creditworthy / Low Risk - the negative class)

    Returns
    -------
    dict
        Dataset container with X, y, metadata.
    """
    df = None
    source = None

    if os.path.exists(filepath):
        try:
            df = _read_csv_flexible(filepath)
            source = f"local file: {filepath}"
        except Exception as exc:
            print(f"[data_loader] Failed reading local German Credit file: {exc}")

    if df is None:
        try:
            openml_obj = fetch_openml(name="credit-g", version=1, as_frame=True, parser="pandas")
            df = openml_obj.frame.copy()
            source = "OpenML credit-g (v1)"
            os.makedirs(os.path.dirname(filepath) or "data", exist_ok=True)
            df.to_csv(filepath, index=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to fetch OpenML 'credit-g': {exc}\n"
                "Please verify internet connection or provide data/german_credit.csv."
            ) from exc

    target_col = "class" if "class" in df.columns else df.columns[-1]

    # Map target reliably across all pandas string/categorical/arrow types
    # 'bad' -> 1 (Default / Adverse credit), 'good' -> 0 (Non-default / Creditworthy)
    raw_target_str = df[target_col].astype(str).str.lower().str.strip()
    mapping = {"bad": 1, "good": 0, "1": 1, "0": 0, "default": 1, "non-default": 0}
    y = raw_target_str.map(mapping).fillna(0).astype(int)

    X = df.drop(columns=[target_col])

    return {
        "X": X,
        "y": y,
        "name": "German Credit (credit-g, OpenML v1)",
        "target_name": target_col,
        "feature_names": list(X.columns),
        "default_rate": float(y.mean()),
        "source": source,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
    }


def load_dataset(name: str = "auto") -> Dict:
    """Load a credit dataset by name with graceful, robust fallback.

    Parameters
    ----------
    name : str, default 'auto'
        'give_me_some_credit', 'german_credit', or 'auto'.
        In 'auto' mode, it attempts Give Me Some Credit first. If unavailable,
        it prints a clear explanation and falls back to German Credit.

    Returns
    -------
    dict
        Dataset dictionary.
    """
    clean_name = name.lower().replace("-", "_").replace(" ", "_")

    if clean_name in ("give_me_some_credit", "gmsc", "primary"):
        return load_give_me_some_credit()

    if clean_name in ("german_credit", "credit_g", "credit-g", "secondary"):
        return load_german_credit()

    if clean_name == "auto":
        if os.environ.get("USE_GMSC", "").lower() in ("1", "true", "yes"):
            ds = load_give_me_some_credit()
            print(
                f"[load_dataset] Successfully loaded PRIMARY: {ds['name']} "
                f"({ds['n_samples']:,} samples, {ds['n_features']} features, "
                f"default rate: {ds['default_rate']:.2%}) from {ds['source']}"
            )
            return ds
        else:
            print(
                "\n[load_dataset] PRIMARY NOTICE: 'Give Me Some Credit' (cs-training.csv) requires manual download "
                "from Kaggle (see README). Falling back gracefully to SECONDARY: German Credit dataset...\n"
            )
            ds = load_german_credit()
            print(
                f"[load_dataset] Successfully loaded SECONDARY: {ds['name']} "
                f"({ds['n_samples']:,} samples, {ds['n_features']} features, "
                f"default rate: {ds['default_rate']:.2%}) from {ds['source']}"
            )
            return ds

    raise ValueError(
        f"Unknown dataset '{name}'. Available: 'give_me_some_credit', 'german_credit', 'auto'."
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split preserving class distribution."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    for ds_name in ("give_me_some_credit", "german_credit"):
        try:
            ds = load_dataset(ds_name)
            print(f"Loaded: {ds['name']} | Samples: {ds['n_samples']:,} | Target rate: {ds['default_rate']:.2%}")
        except Exception as err:
            print(f"Could not load {ds_name}: {err}")
