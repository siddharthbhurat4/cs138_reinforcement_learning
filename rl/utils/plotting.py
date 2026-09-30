"""Plotting helpers. matplotlib is imported lazily so it stays optional."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def moving_average(x: Sequence[float], window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_curves(curves: dict[str, Sequence[float]], window: int = 1, xlabel: str = "Episode",
                ylabel: str = "Return", title: str = "", save_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in curves.items():
        ax.plot(moving_average(values, window), label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
