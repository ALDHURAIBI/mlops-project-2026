"""
package.py — Network Intrusion Detection
MLOps Project 2026 | Chapter 3.2

Purpose
-------
Package the deployed model (params.yaml, deploy.model) for the
BentoML service.

The model is registered in the BentoML Model Store and exported
to an archive that GitHub Actions imports to build the Docker image.

Inputs:
    models/<deploy.model>/model.joblib
    models/<deploy.model>/model_info.json
    params.yaml

Outputs:
    models/network_intrusion_classifier.bentomodel
    models/model_info.json
"""

from pathlib import Path
import json

import bentoml
import bentoml.sklearn
import joblib
import yaml


# ============================================================
# 1. LOAD THE DEPLOYED MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)

MODEL = params["deploy"]["model"]
MODEL_DIR = MODELS_DIR / MODEL

model = joblib.load(MODEL_DIR / "model.joblib")

with open(MODEL_DIR / "model_info.json") as file:
    model_info = json.load(file)

print("Deployed model:", model_info["model_name"])

# The service (serve.py) returns attack and benign probabilities.
if not hasattr(model, "predict_proba"):
    raise ValueError(
        f"{model_info['model_name']} has no predict_proba: "
        "serve.py cannot serve it yet."
    )


# ============================================================
# 2. REGISTER AND EXPORT THE BENTOML MODEL
# ============================================================

# The complete scikit-learn Pipeline is saved, including preprocessing.
bento_model = bentoml.sklearn.save_model(
    "network_intrusion_classifier",
    model,
    metadata={
        "selected_model": model_info["model_name"],
        "feature_columns": model_info["feature_columns"],
    },
)

# Store the exact model version so deployment can load the
# same model rather than an arbitrary latest version.
bento_model_tag = str(bento_model.tag)

print("BentoML model saved:", bento_model_tag)

# Export the BentoML model to the DVC-tracked models directory.
# This makes the model portable across machines.
bento_export_path = MODELS_DIR / "network_intrusion_classifier.bentomodel"

bentoml.models.export_model(
    bento_model_tag,
    str(bento_export_path.resolve())
)

print("BentoML model exported to:", bento_export_path)

# Metadata read by GitHub Actions (exact BentoML tag).
with open(MODELS_DIR / "model_info.json", "w") as file:
    json.dump({**model_info, "bentoml_model_tag": bento_model_tag}, file, indent=2)

print("\nPackaging completed successfully!")
