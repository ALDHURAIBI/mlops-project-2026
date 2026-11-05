
"""
evaluate.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 1.2

Purpose
-------
Evaluate the selected machine learning model on the
holdout test dataset.

Inputs:
    data/prepared/test.csv
    data/prepared/feature_columns.json
    models/baseline_model.joblib
    models/model_info.json
    params.yaml

Outputs:
    evaluation/metrics.json
    evaluation/plots/confusion_matrix.png
    evaluation/plots/precision_recall_curve.png

The outputs will later be tracked by DVC.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
import bentoml.sklearn
import bentoml

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
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
)


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPARED_DIR = PROJECT_ROOT / "data/prepared"
MODEL_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
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

# This model was selected previously using validation data.
# We do not retrain or change its parameters here.

# Load metadata for the selected model.
# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

# Read metadata about the selected model.
with open(MODEL_DIR / "model_info.json") as file:
    model_info = json.load(file)

model_name = model_info["selected_model"]
bento_model_tag = model_info["bentoml_model_tag"]

# Path to the portable BentoML model archive.
bento_archive_path = (
    MODEL_DIR / "network_intrusion_classifier.bentomodel"
)

# Check whether the exact model version exists locally.
try:
    bentoml.models.get(bento_model_tag)
    print("BentoML model found in local Model Store.")

except bentoml.exceptions.NotFound:
    print("Model not found locally. Importing from archive...")

    bentoml.models.import_model(
        str(bento_archive_path.resolve())
    )

    print("BentoML model imported successfully.")

# Load the model using its exact version.
model = bentoml.sklearn.load_model(bento_model_tag)

print("Loaded model:", model_name)
print("BentoML model tag:", bento_model_tag)

# ============================================================
# 3. LOAD HOLDOUT TEST DATA
# ============================================================

# The test partition was created by prepare.py.
# It was not used to train or select the model.

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

# predict_proba() returns class probabilities.
# We use the probability of Attack (class 1) to
# calculate average precision and the PR curve.

attack_class_index = list(model.classes_).index(1)

y_proba = model.predict_proba(X_test)[
    :, attack_class_index
]

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

pr_auc = average_precision_score(
    y_test, y_proba
)

print("\nFinal Test Evaluation")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
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
# DVC will later use this file to compare experiments.

metrics = {
    "model": model_name,
    "test": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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
