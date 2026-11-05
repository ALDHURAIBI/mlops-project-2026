
"""
train.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 1.2

Purpose
-------
Train and compare three baseline machine learning models:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest

The best model is selected using validation PR-AUC
(average precision).

Inputs:
    data/prepared/train.csv
    data/prepared/val.csv
    data/prepared/feature_columns.json
    params.yaml

Outputs:
    models/baseline_model.joblib
    models/model_info.json
    evaluation/validation_model_comparison.csv
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import yaml
import bentoml.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


# ============================================================
# 1. PROJECT PATHS AND PARAMETERS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREPARED_DIR = PROJECT_ROOT / "data/prepared"
MODEL_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

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
# the test set must remain untouched during model selection.

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
# 3. DEFINE BASELINE MODELS
# ============================================================

# Dummy Classifier:
# Predicts the most frequent class.
# It provides a baseline for comparison.

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

# Logistic Regression:
# Missing values are replaced using the training median.
# Features are standardized using training statistics.
#
# Keeping preprocessing inside a Pipeline prevents
# validation information from leaking into training.

lr_config = train_config["logistic_regression"]

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=int(lr_config["max_iter"]),
        class_weight=lr_config["class_weight"],
        random_state=SEED
    ))
])

# Random Forest:
# An ensemble of decision trees that can capture
# nonlinear relationships between network features.
#
# Random Forest does not require feature scaling.
# We only apply missing-value imputation.

rf_config = train_config["random_forest"]

forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=int(rf_config["n_estimators"]),
        max_depth=int(rf_config["max_depth"]),
        min_samples_leaf=int(rf_config["min_samples_leaf"]),
        class_weight=rf_config["class_weight"],
        random_state=SEED,
        n_jobs=-1
    ))
])

models = {
    "Dummy": dummy_model,
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model
}

print("\nModels configured:")
for name in models:
    print("-", name)


# ============================================================
# 4. TRAIN ALL MODELS
# ============================================================

training_times = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    duration = time.perf_counter() - start_time
    training_times[name] = duration

    print(
        f"{name} training completed in {duration:.2f}s"
    )

print("\nAll models trained successfully.")


# ============================================================
# 5. VALIDATION EVALUATION
# ============================================================

# Compare all models using the same validation dataset.
#
# Accuracy alone is misleading for imbalanced datasets.
# We therefore calculate precision, recall, F1-score
# and PR-AUC (average precision).

validation_results = []

for name, model in models.items():

    predictions = model.predict(X_val)

    # Probability that a network flow is an attack.
    probabilities = model.predict_proba(X_val)[:, 1]

    metrics = {
        "model": name,
        "accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "precision": float(
            precision_score(
                y_val, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_val, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_val, predictions, zero_division=0
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_val, probabilities
            )
        ),
        "training_seconds": training_times[name]
    }

    validation_results.append(metrics)

results_df = pd.DataFrame(validation_results)

# Rank models using validation PR-AUC.
selection_metric = train_config["model_selection_metric"]

if selection_metric not in results_df.columns:
    raise ValueError(
        f"Unknown model selection metric: {selection_metric}"
    )

results_df = results_df.sort_values(
    by=selection_metric,
    ascending=False
).reset_index(drop=True)

print("\nValidation results:")
print(results_df.round(4).to_string(index=False))


# ============================================================
# 6. SELECT THE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]
best_model = models[best_model_name]

print("\nBest model:", best_model_name)
print(
    f"Best validation {selection_metric}: "
    f"{results_df.iloc[0][selection_metric]:.4f}"
)

# IMPORTANT:
# Model selection uses validation results only.
# The test dataset will be evaluated in evaluate.py.


# ============================================================
# 7. SAVE TRAINED MODEL AND RESULTS
# ============================================================

# Save the complete fitted model.
# If the model is a Pipeline, the preprocessing steps
# are saved together with the classifier.

model_path = MODEL_DIR / "baseline_model.joblib"

joblib.dump(best_model, model_path)

# Register the selected model in the BentoML Model Store.
# The complete scikit-learn Pipeline is saved, including preprocessing.

bento_model = bentoml.sklearn.save_model(
    "network_intrusion_classifier",
    best_model,
    metadata={
        "selected_model": best_model_name,
        "selection_metric": selection_metric,
        "feature_columns": feature_columns,
    },
)

# Store the exact model version so evaluation and deployment
# can load the same model rather than an arbitrary latest version.
bento_model_tag = str(bento_model.tag)

print("BentoML model saved:", bento_model_tag)

# Export the BentoML model to the DVC-tracked models directory.
# This makes the model portable across machines.

bento_export_path = (
    MODEL_DIR / "network_intrusion_classifier.bentomodel"
)

bentoml.models.export_model(
    bento_model_tag,
    str(bento_export_path.resolve())
)

print("BentoML model exported to:", bento_export_path)

# Save the validation comparison table.
results_path = (
    EVALUATION_DIR / "validation_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

# Save model metadata for reproducibility.
model_info = {
    "bentoml_model_tag": bento_model_tag,
    "selected_model": best_model_name,
    "selection_metric": selection_metric,
    "selection_score": float(
        results_df.iloc[0][selection_metric]
    ),
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
print("Validation results saved to:", results_path)
print("\nTraining pipeline completed successfully!")
