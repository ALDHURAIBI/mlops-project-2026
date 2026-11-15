
"""
evaluate.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 1.2

Purpose
-------
Evaluate one trained model on the holdout test dataset.

Usage: python src/evaluate.py <model>
dvc.yaml runs this script once per model.

Inputs:
    data/prepared/test.csv
    data/prepared/feature_columns.json
    models/<model>/model.joblib
    models/<model>/model_info.json
    params.yaml

Outputs:
    evaluation/<model>/metrics.json
    evaluation/<model>/plots/confusion_matrix.png
    evaluation/<model>/plots/precision_recall_curve.png
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib

# Use a non-interactive backend so plots can also be
# generated automatically on servers and GitHub Actions.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)

# Model definitions: src/models/ (needed to load custom models).
from models import attack_score


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = sys.argv[1]

PREPARED_DIR = PROJECT_ROOT / "data/prepared"
MODEL_DIR = PROJECT_ROOT / "models" / MODEL
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / MODEL
PLOTS_DIR = EVALUATION_DIR / "plots"

EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)

TARGET = params["prepare"]["target"]

print("Starting model evaluation...")


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

# This model was trained by train.py.
# We do not retrain or change its parameters here.

with open(MODEL_DIR / "model_info.json") as file:
    model_info = json.load(file)

model_name = model_info["model_name"]

model = joblib.load(MODEL_DIR / "model.joblib")

print("Loaded model:", model_name)

# ============================================================
# 3. LOAD HOLDOUT TEST DATA
# ============================================================

# The test partition was created by prepare.py.
# It was not used to train or tune the model.

test_df = pd.read_csv(PREPARED_DIR / "test.csv")

with open(PREPARED_DIR / "feature_columns.json") as file:
    feature_columns = json.load(file)

X_test = test_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

y_test = test_df[TARGET].astype(int)

if y_test.nunique() != 2:
    raise ValueError(
        "Test dataset must contain benign and attack flows."
    )

print("\nTest dataset:")
print("Rows:", len(X_test))
print("Features:", len(feature_columns))
print(f"Attack percentage: {y_test.mean() * 100:.2f}%")


# ============================================================
# 4. PREDICT NETWORK ATTACKS
# ============================================================

# predict() returns the predicted class:
# 0 = Benign
# 1 = Attack

y_pred = model.predict(X_test)

# The attack score (probability of Attack for the
# scikit-learn models, reconstruction error for the
# autoencoder) is used to calculate average precision
# and the PR curve.

y_proba = attack_score(model, X_test)

print("\nPrediction completed.")


# ============================================================
# 5. CALCULATE EVALUATION METRICS
# ============================================================

# Accuracy:
# Proportion of correctly classified flows.
#
# Precision:
# Among detected attacks, how many are actual attacks?
#
# Recall:
# Among actual attacks, how many did we detect?
#
# F1-score:
# Harmonic mean of precision and recall.
#
# MCC (Matthews correlation coefficient):
# Uses the four cells of the confusion matrix;
# 1 = perfect, 0 = no better than chance.
#
# Average precision (PR-AUC):
# Summarizes precision-recall performance across
# probability thresholds.

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

mcc = matthews_corrcoef(y_test, y_pred)

pr_auc = average_precision_score(
    y_test, y_proba
)

print("\nFinal Test Evaluation")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

# The confusion matrix shows:
#
# True Negative  : Benign correctly detected
# False Positive : Benign incorrectly flagged as attack
# False Negative : Attack incorrectly classified as benign
# True Positive  : Attack correctly detected

cm = confusion_matrix(
    y_test, y_pred, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Attack"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

ax.set_title(f"Test Confusion Matrix — {model_name}")

plt.tight_layout()

confusion_path = PLOTS_DIR / "confusion_matrix.png"

plt.savefig(confusion_path, dpi=150)
plt.close(fig)

print("\nConfusion matrix:")
print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")

print("Saved:", confusion_path)


# ============================================================
# 7. PRECISION-RECALL CURVE
# ============================================================

# Precision-recall curves are useful for imbalanced
# datasets such as network intrusion detection.
#
# They show the trade-off between detecting attacks
# and producing false alarms at different thresholds.

fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_proba,
    name=model_name,
    ax=ax
)

ax.set_title("Test Precision-Recall Curve")
ax.grid(alpha=0.3)

plt.tight_layout()

pr_curve_path = PLOTS_DIR / "precision_recall_curve.png"

plt.savefig(pr_curve_path, dpi=150)
plt.close(fig)

print("Saved:", pr_curve_path)


# ============================================================
# 8. SAVE METRICS FOR DVC
# ============================================================

# Store all metrics in machine-readable JSON format.
# dvc metrics show compares all models with it.

metrics = {
    "model": model_name,
    "test": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "pr_auc": float(pr_auc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "test_samples": int(len(y_test))
    }
}

metrics_path = EVALUATION_DIR / "metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=2)

print("\nMetrics saved to:", metrics_path)

print("\nEvaluation completed successfully!")
