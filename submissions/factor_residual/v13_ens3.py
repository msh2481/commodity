"""v13 — 3-way ensemble rank-average:
  a) v8a: per-asset residual (PCA K=5) mean-reversion
  b) v10: per-target simple ridge
  c) v8e: no-PCA per-asset mean-reversion on raw returns
Different views on mean-reversion; rank-average for robustness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from v8a_ro_k5 import Model as MA
from v10_per_target import Model as MB
from v8e_nopca import Model as MC


class Model:
    device = "cpu"

    def __init__(self):
        self.models = [MA(), MB(), MC()]

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
