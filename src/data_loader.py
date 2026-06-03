"""Data loading utilities."""
import pandas as pd


def load_data(file_path):
    """Load the marketing campaign dataset from Excel or CSV."""
    file_path = str(file_path)
    if file_path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    raise ValueError("Unsupported file type. Use .xlsx, .xls, or .csv")
