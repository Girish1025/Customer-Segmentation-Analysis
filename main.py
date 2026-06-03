from src.config import DATA_FILE, DROP_COLUMNS, TARGET, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_data
from src.data_cleaning import clean_data, summarize_missing_values
from src.feature_engineering import engineer_features
from src.preprocessing import prepare_logistic_data, prepare_tree_data
from src.model_training import train_logistic_regression, train_decision_tree, train_random_forest, train_xgboost
from src.model_evaluation import evaluate_model, compare_model_metrics


def main():
    # Load and clean data
    raw_data = load_data(DATA_FILE)
    data = clean_data(raw_data, drop_columns=DROP_COLUMNS)
    print("Missing values after cleaning:")
    print(summarize_missing_values(data).head())

    # Feature engineering
    model_data = engineer_features(data, for_eda=False)

    # Logistic Regression preprocessing
    X_train_lr, X_test_lr, y_train_lr, y_test_lr, _ = prepare_logistic_data(
        model_data, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Tree-model preprocessing
    X_train_tree, X_test_tree, y_train_tree, y_test_tree, _ = prepare_tree_data(
        model_data, target=TARGET, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    metrics = []

    # Train and evaluate Logistic Regression
    lr_model = train_logistic_regression(X_train_lr, y_train_lr, RANDOM_STATE)
    lr_metrics, _, _ = evaluate_model(lr_model, X_test_lr, y_test_lr, "Logistic Regression")
    metrics.append(lr_metrics)

    # Train and evaluate Decision Tree
    dt_model = train_decision_tree(X_train_tree, y_train_tree, RANDOM_STATE)
    dt_metrics, _, _ = evaluate_model(dt_model, X_test_tree, y_test_tree, "Decision Tree")
    metrics.append(dt_metrics)

    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train_tree, y_train_tree, RANDOM_STATE)
    rf_metrics, _, _ = evaluate_model(rf_model, X_test_tree, y_test_tree, "Random Forest")
    metrics.append(rf_metrics)

    # Train and evaluate XGBoost
    try:
        xgb_model = train_xgboost(X_train_tree, y_train_tree, RANDOM_STATE)
        xgb_metrics, _, _ = evaluate_model(xgb_model, X_test_tree, y_test_tree, "XGBoost")
        metrics.append(xgb_metrics)
    except ImportError as exc:
        print(exc)

    print("\nModel Performance Comparison:")
    print(compare_model_metrics(metrics))


if __name__ == "__main__":
    main()
