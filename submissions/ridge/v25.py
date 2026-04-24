"""v25: ensemble of v14 across K={20, 30, 40, 50, 60}. Denser coverage."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd
from v21 import _V14Slot


ALPHA = 0.3
K_VALUES = [20, 30, 40, 50, 60]


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._slots = [_V14Slot(ALPHA, k) for k in K_VALUES]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        for s in self._slots:
            s.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = [s.predict(X) for s in self._slots]
        ranks = [(p.rank(axis=1, pct=True) - 0.5) for p in preds]
        return sum(ranks) / len(ranks)
