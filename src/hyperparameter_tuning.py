"""Hyperparameter tuning utilities."""
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def tune_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=2000)
    params = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, params, scoring="f1_weighted", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_decision_tree(X_train, y_train, random_state=42):
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    params = {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, params, scoring="f1_weighted", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
    params = {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, params, scoring="f1_weighted", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_xgboost(X_train, y_train, random_state=42):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    model = XGBClassifier(random_state=random_state, eval_metric="logloss")
    params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid = GridSearchCV(model, params, scoring="f1_weighted", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
