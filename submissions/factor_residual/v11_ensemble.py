"""v11 — Ensemble of per-asset residual (v8a) and per-target (v10).
Average their predictions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from v8a_ro_k5 import Model as ModelA
from v10_per_target import Model as ModelB


class Model:
    device = "cpu"
    w_a = 0.5
    w_b = 0.5

    def __init__(self):
        self.a = ModelA()
        self.b = ModelB()

    def fit(self, X, y):
        self.a.fit(X, y)
        self.b.fit(X, y)

    def predict(self, X):
        pa = self.a.predict(X)
        pb = self.b.predict(X)
        # Rank-standardize each before averaging? The metric is rank-based; if
        # predictors have different scales, plain-avg could be dominated by one.
        # Convert each to cross-sectional rank per date, then avg ranks.
        ra = pa.rank(axis=1, method="average")
        rb = pb.rank(axis=1, method="average")
        combined = self.w_a * ra + self.w_b * rb
        return combined
