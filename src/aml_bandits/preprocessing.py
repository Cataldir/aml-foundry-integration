"""
Classical ML preprocessing with explicit leakage control.
"""

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert target to binary (0/1)."""
    values = series.astype(str).str.lower().str.strip()
    positive = {"yes", "1", "true", "y", "converted", "success"}
    return values.map(lambda v: 1 if v in positive else 0).astype(int)


def preprocess_data(
    raw_df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Preprocess data with leakage control.

    Removes 'duration' column (known only after decision) and separates features/target.

    Args:
        raw_df: Raw dataframe
        target_col: Name of target column

    Returns:
        Tuple of (features_df, target_series, fitted_preprocessor)
    """
    df = raw_df.copy()

    # Remove leakage: duration is known only after the decision
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Normalize target
    y = normalize_target(df[target_col])
    X = df.drop(columns=[target_col])

    # Identify column types
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Build preprocessing pipelines
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    # Combine into column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return X, y, preprocessor
