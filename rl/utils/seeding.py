from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (add torch etc. here later)."""
    random.seed(seed)
    np.random.seed(seed)
