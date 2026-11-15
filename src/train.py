
"""
train.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 1.2

Purpose
-------
Train one model on the training data and report its
validation metrics.

Usage: python src/train.py <model>
<model> is a file of src/models/ (random_forest, autoencoder...).
dvc.yaml runs this script once per model.

Inputs:
    data/prepared/train.csv
    data/prepared/val.csv
    data/prepared/feature_columns.json
    params.yaml

Outputs:
    models/<model>/model.joblib
    models/<model>/model_info.json
    evaluation/<model>/validation_metrics.json
"""

from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    matthews_corrcoef,
)

# Model definitions: src/models/, one file per model.
from models import attack_score, build_model, display_name


# ============================================================
# 1. PROJECT PATHS AND PARAMETERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = sys.argv[1]

PREPARED_DIR = PROJECT_ROOT / "data/prepared"
MODEL_DIR = PROJECT_ROOT / "models" / MODEL
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / MODEL

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# Read experiment configuration from params.yaml.
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)

prepare_config = params["prepare"]
train_config = params["train"]

SEED = int(prepare_config["seed"])
TARGET = prepare_config["target"]

print("Starting model training...")
print("Random seed:", SEED)


# ============================================================
# 2. LOAD PREPARED DATASETS
# ============================================================

# The preparation script already created group-aware
# training, validation and test datasets.
#
# IMPORTANT: We do not load test.csv here because
# the test set must remain untouched while models are tuned.

train_df = pd.read_csv(PREPARED_DIR / "train.csv")
val_df = pd.read_csv(PREPARED_DIR / "val.csv")

# Load the original feature column order.
# This prevents mismatches between training and inference.
with open(PREPARED_DIR / "feature_columns.json") as file:
    feature_columns = json.load(file)

X_train = train_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
y_train = train_df[TARGET].astype(int)

X_val = val_df[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)
y_val = val_df[TARGET].astype(int)

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Number of features: {len(feature_columns)}")

print("\nTraining class distribution:")
print(y_train.value_counts())


# ============================================================
# 3. BUILD THE MODEL
# ============================================================

# Definition in src/models/<model>.py,
# settings in params.yaml (train.<model>).

MODEL_NAME = display_name(MODEL)

model = build_model(MODEL, train_config, SEED)

print("\nModel:", MODEL_NAME)


# ============================================================
# 4. TRAIN THE MODEL
# ============================================================

start_time = time.perf_counter()

model.fit(X_train, y_train)

training_seconds = time.perf_counter() - start_time

print(f"{MODEL_NAME} training completed in {training_seconds:.2f}s")


# ============================================================
# 5. VALIDATION EVALUATION
# ============================================================

# Accuracy alone is misleading for imbalanced datasets.
# We therefore calculate precision, recall, F1-score,
# MCC and PR-AUC (average precision).
#
# The validation set is used to tune the model settings.
# The test dataset will be evaluated in evaluate.py.

predictions = model.predict(X_val)

# Attack score: probability of the attack class, or
# reconstruction error for the autoencoder.
scores = attack_score(model, X_val)

validation = {
    "accuracy": float(accuracy_score(y_val, predictions)),
    "precision": float(
        precision_score(y_val, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, predictions, zero_division=0)
    ),
    "f1": float(f1_score(y_val, predictions, zero_division=0)),
    "mcc": float(matthews_corrcoef(y_val, predictions)),
    "pr_auc": float(average_precision_score(y_val, scores)),
}

print("\nValidation results:")
for name, value in validation.items():
    print(f"{name:9}: {value:.4f}")


# ============================================================
# 6. SAVE TRAINED MODEL AND RESULTS
# ============================================================

# Save the complete fitted model.
# If the model is a Pipeline, the preprocessing steps
# are saved together with the classifier.
# The deployed model is packaged for BentoML by package.py.

model_path = MODEL_DIR / "model.joblib"

joblib.dump(model, model_path)

# Save the validation metrics for DVC (dvc metrics show).
validation_path = EVALUATION_DIR / "validation_metrics.json"

with open(validation_path, "w") as file:
    json.dump({"model": MODEL_NAME, "validation": validation}, file, indent=2)

# Save model metadata for reproducibility.
# No timing here: the folder must be identical when the
# model is, so that DVC does not rerun the next stages.
model_info = {
    "model": MODEL,
    "model_name": MODEL_NAME,
    "feature_count": len(feature_columns),
    "feature_columns": feature_columns,
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "seed": SEED
}

with open(MODEL_DIR / "model_info.json", "w") as file:
    json.dump(model_info, file, indent=2)

print("\nModel saved to:", model_path)
print("Model metadata saved to:", MODEL_DIR / "model_info.json")
print("Validation metrics saved to:", validation_path)
print("\nTraining completed successfully!")
