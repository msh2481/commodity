"""v23: ensemble of v14 K={30, 45} and v6 (no feature selection).

Mix feature-selected and all-features models for diversity.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from v21 import _V14Slot
from v6 import Model as ModelV6


ALPHA = 0.3


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._s30 = _V14Slot(ALPHA, 30)
        self._s45 = _V14Slot(ALPHA, 45)
        self._v6 = ModelV6()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._s30.fit(X, y)
        self._s45.fit(X, y)
        self._v6.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = [self._s30.predict(X), self._s45.predict(X), self._v6.predict(X)]
        ranks = [(p.rank(axis=1, pct=True) - 0.5) for p in preds]
        return sum(ranks) / len(ranks)
