"""
base.py — Network Intrusion Detection
MLOps Project 2026

Interface of the custom models.

The scikit-learn models (Dummy, Logistic Regression, Random Forest)
already have fit(X, y) and predict(X). A custom model subclasses
Detector: it defines fit(), which must set a threshold, and
attack_score(); predict() then flags the flows whose score is
above the threshold. Train and replay code can use any model the
same way.

The Random Forest stays a plain scikit-learn Pipeline on purpose:
the BentoML service loads it without importing this package.
"""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class Detector(BaseEstimator, ABC):

    @abstractmethod
    def fit(self, X, y=None):
        """Learn from flows X (labels y: 1 = attack). Sets self.threshold_."""

    @abstractmethod
    def attack_score(self, X):
        """One score per flow: the higher, the more suspicious."""

    def predict(self, X):
        return (self.attack_score(X) > self.threshold_).astype(int)
