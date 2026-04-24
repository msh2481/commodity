"""v13: rank-ensemble of v6 (rank-y) + v11 (bagging) + v12 (advanced stats).

Fit all three and average per-date ranks.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from v6 import Model as ModelV6
from v11 import Model as ModelV11
from v12 import Model as ModelV12


W_V6 = 0.5
W_V11 = 0.2
W_V12 = 0.3


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._m6 = ModelV6()
        self._m11 = ModelV11()
        self._m12 = ModelV12()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._m6.fit(X, y)
        self._m11.fit(X, y)
        self._m12.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p6 = self._m6.predict(X)
        p11 = self._m11.predict(X)
        p12 = self._m12.predict(X)
        r6 = p6.rank(axis=1, pct=True) - 0.5
        r11 = p11.rank(axis=1, pct=True) - 0.5
        r12 = p12.rank(axis=1, pct=True) - 0.5
        return W_V6 * r6 + W_V11 * r11 + W_V12 * r12
