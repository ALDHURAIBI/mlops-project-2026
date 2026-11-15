"""
logistic_regression.py — Network Intrusion Detection
MLOps Project 2026

Logistic Regression, the linear baseline.

Missing values are replaced using the training median and
features are standardized using training statistics. Keeping
preprocessing inside a Pipeline prevents validation information
from leaking into training.
Settings: params.yaml, train.logistic_regression.
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NAME = "Logistic Regression"


def build(config, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=int(config["max_iter"]),
            class_weight=config["class_weight"],
            random_state=seed
        ))
    ])
