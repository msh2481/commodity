"""v17 — 7-way ensemble: more diverse variants."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from v8a_ro_k5 import Model as _Ma
from v10_per_target import Model as _Mb
from v8e_nopca import Model as _Mc
from v15_pt_multiscale import Model as _Md


class _PT_a10(_Mb):
    alpha = 10.0


class _PT_l10(_Mb):
    L = 10


class _RO_k10(_Ma):
    K = 10


class _MS_a10(_Md):
    alpha = 10.0


class Model:
    device = "cpu"

    def __init__(self):
        self.models = [_Ma(), _Mb(), _Mc(), _Md(), _PT_a10(), _PT_l10(), _RO_k10(), _MS_a10()]

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
