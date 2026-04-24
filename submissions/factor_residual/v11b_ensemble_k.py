"""v11b — Ensemble across multiple K values of residual-only per-asset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from v7c_resid_only import Model as _Base


class _K2(_Base):
    K = 2; alpha_R = 1.0


class _K5(_Base):
    K = 5; alpha_R = 1.0


class _K10(_Base):
    K = 10; alpha_R = 1.0


class _K20(_Base):
    K = 20; alpha_R = 1.0


class Model:
    device = "cpu"

    def __init__(self):
        self.models = [_K2(), _K5(), _K10(), _K20()]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):
        ranks = None
        for m in self.models:
            p = m.predict(X)
            r = p.rank(axis=1, method="average")
            ranks = r if ranks is None else ranks + r
        return ranks / len(self.models)
