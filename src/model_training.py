"""Model training functions."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def train_logistic_regression(X_train, y_train, random_state=42):
    model = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, random_state=42):
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Install it with: pip install xgboost")
    model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)
    return model
