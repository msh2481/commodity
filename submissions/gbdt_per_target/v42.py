"""v42: v38_tune pipeline + cross-sectional rank(y) target transform.

Replaces y[t] with the per-date rank of y[t] among all kept targets, scaled
to [-1, 1]. Since the competition metric is Spearman rank correlation per
date across targets, training on ranks directly aligns the XGBoost loss
with the evaluation target.

Loads hparams from _tune/hparams.json (same file as v38_tune).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata

from v38_tune import Model as _V38Tune


def cross_sectional_rank(y_arr: np.ndarray) -> np.ndarray:
    """Per-row rank of finite entries, scaled to [-1, 1]. NaN → NaN."""
    out = np.full_like(y_arr, np.nan, dtype=np.float32)
    for i in range(y_arr.shape[0]):
        row = y_arr[i]
        mask = np.isfinite(row)
        n = int(mask.sum())
        if n < 2:
            continue
        ranks = rankdata(row[mask], method="average")
        out[i, mask] = ((ranks - 1) / (n - 1) * 2.0 - 1.0).astype(np.float32)
    return out


class Model(_V38Tune):
    def fit(self, X, y):
        target_cols = list(y.columns)
        keep_idx = [j for j, t in enumerate(target_cols) if self._is_kept(t)]
        kept_cols = [target_cols[j] for j in keep_idx]

        y_kept = y[kept_cols].to_numpy(dtype=np.float32)
        y_kept_ranked = cross_sectional_rank(y_kept)

        y_new = y.copy()
        y_new[kept_cols] = y_kept_ranked
        super().fit(X, y_new)
