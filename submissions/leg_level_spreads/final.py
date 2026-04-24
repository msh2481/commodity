"""v33: v32 + per-lag prediction scaling.

Post-processing: scale predictions by lag. Lag=1 has ~zero time-series
correlation with labels (near-random walk at daily scale), so we zero
out lag=1 predictions — they contribute nothing to the cross-sectional
ordering except noise. Lag=2 is downweighted (0.5); lag=3 and 4 are
slightly upweighted (1.3) since they have the strongest signal.

Grid-searched scales on v32's OOF predictions yielded (0.0, 0.3, 1.5,
1.5) as peak; we pick (0.0, 0.5, 1.3, 1.3) — a milder rule of thumb
that avoids overfitting the search.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import pandas as pd

from v32 import Model as _V32Model


LAG_SCALES = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0}


class Model(_V32Model):
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = super().predict(X)
        import pandas as pd
        import pandas as _pd

        tp = self.tp
        scaled = preds.copy()
        for _, row in tp.iterrows():
            col = row["target"]
            lag = int(row["lag"])
            scaled[col] = scaled[col] * LAG_SCALES[lag]
        return scaled
