
"""
serve.py — Network Intrusion Detection API
MLOps Project 2026 | Chapter 3.2

Serve the selected scikit-learn model with BentoML.
Input: 39 named network-flow features.
Output: Attack/Benign prediction and probabilities.
"""

import json
from pathlib import Path
from bentoml.exceptions import UnprocessableEntity
import bentoml
import bentoml.sklearn
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

with open(MODEL_DIR / "model_info.json") as file:
    model_info = json.load(file)

MODEL_TAG = model_info["bentoml_model_tag"]
FEATURE_COLUMNS = model_info["feature_columns"]

ARCHIVE_PATH = (
    MODEL_DIR / "network_intrusion_classifier.bentomodel"
)


# Restore the model if the local Model Store is empty.
try:
    bentoml.models.get(MODEL_TAG)
except bentoml.exceptions.NotFound:
    bentoml.models.import_model(str(ARCHIVE_PATH.resolve()))


# Define the API input format.
class PredictionRequest(BaseModel):
    features: dict[str, float | None] = Field(
        description="Network-flow features keyed by their exact training column names"
    )


# Define the API output format.
class PredictionResponse(BaseModel):
    prediction: str
    predicted_class: int
    attack_probability: float
    benign_probability: float


@bentoml.service(
    name="network_intrusion_service",
    resources={"cpu": "1"},
    traffic={"timeout": 30},
)
class NetworkIntrusionService:

    def __init__(self):
        self.model = bentoml.sklearn.load_model(MODEL_TAG)

    @bentoml.api
    def predict(self, request: PredictionRequest) -> PredictionResponse:

        features = request.features

        # Validate feature names.
        missing = set(FEATURE_COLUMNS) - set(features)
        unexpected = set(features) - set(FEATURE_COLUMNS)

        if missing or unexpected:
            raise UnprocessableEntity(
                f"Missing features: {sorted(missing)}; "
                f"Unexpected features: {sorted(unexpected)}"
            )

        # Reconstruct the exact training feature order.
        input_df = pd.DataFrame(
            [[features[name] for name in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS,
        )

        # Apply the same non-finite value handling as training.
        input_df = input_df.replace(
            [np.inf, -np.inf], np.nan
        )

        # Predict using the saved preprocessing + classifier.
        predicted_class = int(self.model.predict(input_df)[0])

        probabilities = self.model.predict_proba(input_df)[0]
        class_probabilities = dict(
            zip(self.model.classes_, probabilities)
        )

        attack_probability = float(class_probabilities[1])
        benign_probability = float(class_probabilities[0])

        return PredictionResponse(
            prediction="Attack" if predicted_class == 1 else "Benign",
            predicted_class=predicted_class,
            attack_probability=attack_probability,
            benign_probability=benign_probability,
        )
