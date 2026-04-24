"""v22 — Ensemble incl. a proper factor+residual model to honor the thread approach."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from v8a_ro_k5 import Model as MA
from v10_per_target import Model as MB
from v8e_nopca import Model as MC
from v15_pt_multiscale import Model as MD
from v19_pt_vol_feats import Model as ME
from v6a_pa_a1 import Model as MF  # factor + per-asset residual (α_R=1, K=10)


class Model:
    device = "cpu"

    def __init__(self):
        self.models = [MA(), MB(), MC(), MD(), ME(), MF()]

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
