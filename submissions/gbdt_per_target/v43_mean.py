"""v43_mean: quantile-regression XGBoost, rank targets by mean of quantile preds."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v43_quantile import Model as _QModel


class Model(_QModel):
    AGG_METHOD = "mean"
