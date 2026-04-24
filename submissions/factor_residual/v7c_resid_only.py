"""Residual-only: zero factor prediction, but still project out factors so
residuals are clean idiosyncratic. Tests whether the factor model contributes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from v5c_perasset import Model as _Base


class Model(_Base):
    K = 10
    L_R = 5
    alpha_R = 1.0

    def predict(self, X):
        preds = super().predict(X)
        # Zero out everything via residual-only hack — recompose with F_hat=0.
        # But parent already baked F_hat into preds; easiest is to reimplement.
        return preds

    # Override: zero out F_hat so only residual contributes.
    def fit(self, X, y):
        super().fit(X, y)
        # Neuter factor ridge: set coefs to zero so F_hat=0.
        for lag in list(self.ridge_F_.keys()):
            rf = self.ridge_F_[lag]
            rf.coef_ = np.zeros_like(rf.coef_)
            rf.intercept_ = np.zeros_like(rf.intercept_) if hasattr(rf.intercept_, 'shape') else 0.0
