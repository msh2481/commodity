"""v24 — 6-way ensemble: v20 + XGBoost (v23). XGBoost has lower std_rho,
providing structural diversity."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from v8a_ro_k5 import Model as MA
from v10_per_target import Model as MB
from v8e_nopca import Model as MC
from v15_pt_multiscale import Model as MD
from v19_pt_vol_feats import Model as ME
from v23_xgb_pt import Model as MF


class Model:
    device = "cpu"
    cpus_per_fold = 16

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
