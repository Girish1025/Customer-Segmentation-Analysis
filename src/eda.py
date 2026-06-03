"""Exploratory data analysis utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def plot_target_distribution(df, target="response"):
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x=target)
    total = len(df)
    for p in ax.patches:
        count = p.get_height()
        ax.annotate(f"{count} ({count / total * 100:.1f}%)",
                    (p.get_x() + p.get_width() / 2, count),
                    ha="center", va="bottom")
    plt.title("Target Variable Distribution")
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, columns):
    ncols = 3
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[len(columns):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def response_rate_by_group(df, group_col, target="response"):
    result = df.groupby(group_col)[target].agg(["count", "sum", "mean"]).reset_index()
    result = result.rename(columns={"sum": "accepted", "mean": "response_rate"})
    result["response_rate_percent"] = (result["response_rate"] * 100).round(2)
    return result.sort_values("response_rate_percent", ascending=False)


def plot_response_rate(df, group_col, target="response"):
    rates = response_rate_by_group(df, group_col, target)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=rates, x=group_col, y="response_rate_percent")
    plt.title(f"Response Rate by {group_col}")
    plt.ylabel("Response Rate (%)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
    return rates


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def correlation_analysis(df):
    numeric_features = df.select_dtypes(include=["float64", "int64", "int32", "float32"]).columns
    corr_matrix = df[numeric_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = upper.stack().reset_index()
    high_corr.columns = ["feature_1", "feature_2", "correlation"]
    return corr_matrix, high_corr[high_corr["correlation"] > 0.8]
