"""
dummy.py — Network Intrusion Detection
MLOps Project 2026

Dummy Classifier: predicts the most frequent class.
It gives the lower bound every real model must beat.
Settings: params.yaml, train.dummy.
"""

from sklearn.dummy import DummyClassifier

NAME = "Dummy"


def build(config, seed):
    return DummyClassifier(strategy=config["strategy"])
