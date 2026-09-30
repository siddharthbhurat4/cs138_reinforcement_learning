"""Action / observation spaces. They describe *what* an env accepts and emits."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Space(ABC):
    def __init__(self, seed: Any = None):
        self.seed(seed)

    def seed(self, seed: Any = None) -> None:
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self) -> Any: ...

    @abstractmethod
    def contains(self, x: Any) -> bool: ...


class Discrete(Space):
    """Integers {0, 1, ..., n-1}."""

    def __init__(self, n: int, seed: Any = None):
        self.n = int(n)
        super().__init__(seed)

    def sample(self) -> int:
        return int(self.rng.integers(self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box(Space):
    """Continuous (or bounded integer) n-dimensional box."""

    def __init__(self, low, high, shape: tuple | None = None, dtype=np.float32, seed: Any = None):
        if shape is None:
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = self.low.shape
        else:
            self.shape = tuple(shape)
            self.low = np.full(self.shape, low, dtype=dtype)
            self.high = np.full(self.shape, high, dtype=dtype)
        self.dtype = dtype
        super().__init__(seed)

    def sample(self) -> np.ndarray:
        low = np.where(np.isfinite(self.low), self.low, -1e6)
        high = np.where(np.isfinite(self.high), self.high, 1e6)
        return self.rng.uniform(low, high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"Box(shape={self.shape}, dtype={np.dtype(self.dtype).name})"
