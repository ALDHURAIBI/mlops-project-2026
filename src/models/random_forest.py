"""
random_forest.py — Network Intrusion Detection
MLOps Project 2026

Random Forest, the model of the project.

An ensemble of decision trees that can capture nonlinear
relationships between network features. It does not need
feature scaling: missing values are imputed only.
Settings: params.yaml, train.random_forest.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NAME = "Random Forest"


def build(config, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            class_weight=config["class_weight"],
            random_state=seed,
            n_jobs=-1
        ))
    ])
