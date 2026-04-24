"""v26 — XGBoost per target, deeper trees and more estimators."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v23_xgb_pt import Model as _Base


class Model(_Base):
    L = 10
    n_estimators = 100
    max_depth = 5
    learning_rate = 0.05
