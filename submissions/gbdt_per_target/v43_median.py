"""v43_median: quantile-regression XGBoost, rank targets by predicted 0.5-quantile."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v43_quantile import Model as _QModel


class Model(_QModel):
    AGG_METHOD = "median"
