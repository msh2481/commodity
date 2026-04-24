"""v1: zero leg-return predictor (pipeline sanity).

Predicts leg_return = 0 for every (asset, lag). Compose into target
predictions via the leg -> target mapping. Expected Sharpe ~0.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd

from utils import LAGS, compose_targets, load_target_pairs, unique_legs


class Model:
    device = "cpu"

    def __init__(self) -> None:
        self._tp = load_target_pairs()
        self._legs = unique_legs(self._tp)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        return None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = pd.MultiIndex.from_product(
            [self._legs, LAGS], names=["asset", "lag"]
        )
        leg_preds = pd.DataFrame(
            np.zeros((len(X), len(cols))), index=X.index, columns=cols
        )
        return compose_targets(leg_preds, self._tp)
