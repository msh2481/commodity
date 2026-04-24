import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v12_per_target_rich import Model as _Base


class Model(_Base):
    L = 5
    alpha = 10.0
