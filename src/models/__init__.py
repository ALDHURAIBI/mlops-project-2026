"""
Model definitions of the project, one file per model.

    dummy.py                 most frequent class (lower bound)
    logistic_regression.py   linear baseline
    random_forest.py         model of the project
    autoencoder.py           anomaly detector trained on benign flows
    base.py                  interface of the custom models

Each file defines NAME and build(config, seed). The pipeline trains,
evaluates and replays every model listed in dvc.yaml; a model is
loaded by its name only, so changing one file only reruns the
stages of that model. To add a model: a new file here, its settings
under train in params.yaml, and its name in the lists of dvc.yaml.

Every model follows the scikit-learn interface: fit(X, y), then
predict(X) with 1 = attack and 0 = benign.

This folder holds code. The trained models are written by train.py
to models/<name>/ at the project root, tracked by DVC.
"""

import importlib


def load(name):
    """Module src/models/<name>.py."""
    return importlib.import_module(f"{__name__}.{name}")


def build_model(name, train_config, seed):
    """Untrained model, with its settings from params.yaml (train.<name>)."""
    return load(name).build(train_config[name], seed)


def display_name(name):
    return load(name).NAME


def attack_score(model, X):
    """Attack score per flow: probability of the attack class for the
    scikit-learn models, reconstruction error for the autoencoder."""
    if hasattr(model, "attack_score"):
        return model.attack_score(X)
    return model.predict_proba(X)[:, list(model.classes_).index(1)]
