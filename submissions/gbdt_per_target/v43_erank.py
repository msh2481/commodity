"""v43_erank: quantile-regression XGBoost, rank targets by expected rank.

For each date, flatten the (N_kept, M_quantiles) matrix, global-rank all
entries, and score each target by the mean rank over its M quantile samples.

This is the rearrangement-inequality-optimal score when each target's marginal
posterior is represented by its empirical quantile distribution, and the
metric is per-date Spearman rank correlation across targets.
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v43_quantile import Model as _QModel


class Model(_QModel):
    AGG_METHOD = "erank"
