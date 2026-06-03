"""Data cleaning functions."""
import re
import pandas as pd


def clean_column_names(df):
    """Convert columns to lowercase snake-style names."""
    df = df.copy()
    clean_cols = []
    for col in df.columns:
        col = str(col).strip().lower()
        col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
        col = re.sub(r"_+", "_", col).strip("_")
        clean_cols.append(col)
    df.columns = clean_cols
    return df


def clean_data(df, drop_columns=None):
    """Clean raw marketing campaign data."""
    df = clean_column_names(df)
    df = df.replace("", pd.NA).drop_duplicates()

    if drop_columns is None:
        drop_columns = []
    existing_drop_cols = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=existing_drop_cols, errors="ignore")

    if "income" in df.columns:
        df["income"] = df["income"].fillna(df["income"].median())

    return df


def summarize_missing_values(df):
    """Return missing-value counts and percentages."""
    missing = df.isna().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing["missing_percent"] = (missing["missing_count"] / len(df) * 100).round(2)
    return missing.sort_values("missing_count", ascending=False)
