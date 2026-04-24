import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v7c_resid_only import Model as _Base


class Model(_Base):
    K = 5
    L_R = 5
    alpha_R = 10.0
