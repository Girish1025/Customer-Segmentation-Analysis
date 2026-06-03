"""SHAP model interpretation utilities."""
import shap
import matplotlib.pyplot as plt


def run_tree_shap_analysis(model, X_test, plot_type="bar"):
    """Generate SHAP summary plot for a tree-based model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type=plot_type)
    plt.tight_layout()
    return shap_values
