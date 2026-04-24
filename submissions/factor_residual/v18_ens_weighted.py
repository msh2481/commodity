"""v18 — Weighted 4-way ensemble. Downweight the regime-sensitive v15."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from v8a_ro_k5 import Model as MA
from v10_per_target import Model as MB
from v8e_nopca import Model as MC
from v15_pt_multiscale import Model as MD


class Model:
    device = "cpu"
    weights = (1.0, 1.0, 1.0, 0.5)

    def __init__(self):
        self.models = [MA(), MB(), MC(), MD()]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict(self, X):
        ranks = None
        wsum = 0.0
        for m, w in zip(self.models, self.weights):
            p = m.predict(X)
            r = p.rank(axis=1, method="average") * w
            ranks = r if ranks is None else ranks + r
            wsum += w
        return ranks / wsum
