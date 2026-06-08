"""
Data loading with Kaggle-first priority, local CSV fallback, and OpenML reproducibility.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.datasets import fetch_openml


def _try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    """Try reading a CSV with multiple separators."""
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 2:
                return df
        except Exception:
            continue
    return None


def _find_target_column(columns: Sequence[str]) -> Optional[str]:
    """Find common target column names in dataset."""
    lowered = {c.lower(): c for c in columns}
    candidates = [
        "y",
        "target",
        "label",
        "subscribed",
        "conversion",
        "converted",
        "response",
        "class",
    ]
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _try_load_from_kaggle(repo_root: Path) -> Optional[Tuple[pd.DataFrame, str, str]]:
    """Attempt to load dataset from Kaggle API."""
    kaggle_dir = repo_root / "tmp" / "7mlet" / "kaggle_bank_marketing"
    kaggle_dir.mkdir(parents=True, exist_ok=True)

    csv_candidates = sorted(kaggle_dir.rglob("*.csv"))

    if not csv_candidates:
        datasets = [
            "henriqueyamahata/bank-marketing",
            "tunguz/bank-marketing-data-set",
            "dharmik34/bank-term-deposit-subscription",
        ]
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            for ds in datasets:
                try:
                    api.dataset_download_files(
                        ds, path=str(kaggle_dir), unzip=True, quiet=True
                    )
                except Exception:
                    continue

            csv_candidates = sorted(kaggle_dir.rglob("*.csv"))
        except Exception as ex:
            print(f"Kaggle load skipped: {ex}")
            return None

    for path in csv_candidates:
        df = _try_read_csv(path)
        if df is None:
            continue
        target_col = _find_target_column(df.columns)
        if target_col is None:
            continue
        return df, target_col, f"kaggle dataset file: {path.as_posix()}"

    return None


def load_bank_marketing_dataset() -> Tuple[pd.DataFrame, str, str]:
    """
    Load bank marketing dataset with Kaggle priority and fallback.

    Returns:
        Tuple of (dataframe, target_column_name, provenance_string)
    """
    repo_root = Path.cwd()

    # Try Kaggle first
    kaggle_loaded = _try_load_from_kaggle(repo_root)
    if kaggle_loaded is not None:
        return kaggle_loaded

    # Try local CSV files
    candidates = [
        repo_root / "tmp" / "7mlet" / "bank-additional-full.csv",
        repo_root / "tmp" / "7mlet" / "bank-full.csv",
        repo_root / "tmp" / "7mlet" / "bank.csv",
        repo_root / "data" / "assets" / "bank-additional-full.csv",
        repo_root / "data" / "assets" / "bank-full.csv",
    ]

    for path in candidates:
        df = _try_read_csv(path)
        if df is None:
            continue
        target_col = _find_target_column(df.columns)
        if target_col is None:
            continue
        return df, target_col, f"local CSV: {path.as_posix()}"

    # Fallback to OpenML
    try:
        bunch = fetch_openml(
            name="bank-marketing", version=1, as_frame=True, parser="auto"
        )
    except TypeError:
        bunch = fetch_openml(name="bank-marketing", version=1, as_frame=True)

    df = bunch.frame.copy()
    target_col = _find_target_column(df.columns)
    if target_col is None:
        raise ValueError("Unable to identify target column in OpenML bank-marketing dataset.")
    return df, target_col, "OpenML: bank-marketing (version=1)"
