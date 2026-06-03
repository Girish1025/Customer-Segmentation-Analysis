"""Preprocessing functions for model training."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE


def split_features_target(df, target="response"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def prepare_logistic_data(df, target="response", test_size=0.2, random_state=42):
    X, y = split_features_target(df, target)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["float64", "int64", "int32", "float32"]).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    if numeric_cols:
        X_train.loc[:, numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    smote = SMOTE(random_state=random_state)
    X_train_rs, y_train_rs = smote.fit_resample(X_train, y_train)
    return X_train_rs, X_test, y_train_rs, y_test, scaler


def prepare_tree_data(df, target="response", test_size=0.2, random_state=42):
    X, y = split_features_target(df, target)
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    smote = SMOTE(random_state=random_state)
    X_train_rs, y_train_rs = smote.fit_resample(X_train, y_train)
    return X_train_rs, X_test, y_train_rs, y_test, encoders
