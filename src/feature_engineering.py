"""Feature engineering for campaign response prediction."""
from datetime import datetime
import pandas as pd


def group_marital_status(value):
    if value in ["Single", "Divorced", "Widow", "Alone"]:
        return "Living alone"
    if value in ["Together", "Married"]:
        return "Not Living alone"
    return "Other"


def create_age_group(age):
    if age < 25:
        return "Youth"
    if age <= 34:
        return "Young Adult"
    if age <= 49:
        return "Adult"
    if age <= 64:
        return "Senior"
    return "Elderly"


def create_income_group(income):
    if income < 40000:
        return "Low"
    if income <= 80000:
        return "Medium"
    return "High"


def create_recency_group(recency):
    bins = [0, 15, 45, 75, 100, 150]
    labels = ["Active Loyalists", "Warm Leads", "Fading Customers", "At-Risk Segment", "Lapsed Customers"]
    return pd.cut(pd.Series(recency), bins=bins, labels=labels, right=True).iloc[0]


def engineer_features(df, for_eda=False):
    """Create model-ready and EDA-ready customer features."""
    df = df.copy()
    current_year = datetime.now().year

    if "year_birth" in df.columns:
        df["age"] = current_year - df["year_birth"]
        if not for_eda:
            df = df.drop(columns=["year_birth"])

    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].apply(group_marital_status)

    if {"kidhome", "teenhome"}.issubset(df.columns):
        df["totalkids"] = df["kidhome"] + df["teenhome"]
        if not for_eda:
            df = df.drop(columns=["kidhome", "teenhome"])

    if "dt_customer" in df.columns:
        df["dt_customer"] = pd.to_datetime(df["dt_customer"], errors="coerce")
        today = pd.to_datetime("today")
        df["years_enrolled"] = ((today - df["dt_customer"]).dt.days / 365).astype("Int64")
        df["years_enrolled"] = df["years_enrolled"].fillna(df["years_enrolled"].median()).astype(int)
        if not for_eda:
            df = df.drop(columns=["dt_customer"])

    if for_eda and "age" in df.columns:
        df["age_group"] = df["age"].astype(float).apply(create_age_group)
    if for_eda and "income" in df.columns:
        df["income_group"] = df["income"].apply(create_income_group)
    if for_eda and "recency" in df.columns:
        df["recency_group"] = df["recency"].apply(create_recency_group)

    return df
