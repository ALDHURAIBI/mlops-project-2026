"""
autoencoder.py — Network Intrusion Detection
MLOps Project 2026

Autoencoder, an anomaly detector.

It learns to reconstruct benign flows only; the attack score is
the reconstruction error. It can flag an attack family it has
never seen, but it also flags unusual benign traffic.

Training: up to train_cap benign flows, 80 % to fit the network,
20 % held out to set the threshold so that a share
false_positive_target of them is flagged.
Settings: params.yaml, train.autoencoder.
"""

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .base import Detector

NAME = "Autoencoder"


class Autoencoder(Detector):

    def __init__(self, hidden_layer_sizes=(32, 8, 32), max_iter=200,
                 train_cap=200_000, false_positive_target=0.01,
                 random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.train_cap = train_cap
        self.false_positive_target = false_positive_target
        self.random_state = random_state

    def _prepare(self, X):
        # Rates can be infinite (NaN after prepare.py) or missing.
        x = np.nan_to_num(np.asarray(X, dtype="float64"))
        return np.log1p(np.clip(x, 0, None))

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        benign = X if y is None else X[np.asarray(y) == 0]
        benign = benign.sample(
            min(self.train_cap, len(benign)),
            random_state=self.random_state
        )
        cut = int(0.8 * len(benign))

        self.scaler_ = StandardScaler()
        self.net_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=self.random_state
        )
        z = self.scaler_.fit_transform(self._prepare(benign.iloc[:cut]))
        self.net_.fit(z, z)

        self.threshold_ = np.quantile(
            self.attack_score(benign.iloc[cut:]),
            1 - self.false_positive_target
        )
        return self

    def attack_score(self, X):
        z = self.scaler_.transform(self._prepare(X))
        return ((z - self.net_.predict(z)) ** 2).mean(axis=1)


def build(config, seed):
    return Autoencoder(
        hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
        max_iter=int(config["max_iter"]),
        train_cap=int(config["train_cap"]),
        false_positive_target=float(config["false_positive_target"]),
        random_state=seed
    )
