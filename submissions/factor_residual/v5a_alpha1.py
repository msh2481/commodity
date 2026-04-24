import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3 import Model as _BaseModel


class Model(_BaseModel):
    K = 10
    alpha_R = 1.0
