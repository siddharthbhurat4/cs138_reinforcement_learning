"""Plotting for the binary-increment bandit. Reuses plot_bandit_results from the spiral plotting file."""
import os

import matplotlib.pyplot as plt
import numpy as np

from homework1.plotting_spiral_randomwalk import plot_bandit_results  # noqa: F401  (re-exported)


def plot_action_means(means, max_steps=2048, title="True action means (binary counter)", save_path=None):
    """Heatmap of the true means: one row per action, black = mean is high (bit is 1)."""
    means = np.asarray(means)[:max_steps]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.imshow(means.T, aspect="auto", cmap="Greys", interpolation="nearest", origin="lower")
    ax.set_yticks(range(means.shape[1]))
    ax.set_yticklabels([f"action {k}" for k in range(means.shape[1])])
    ax.set_xlabel("Step")
    ax.set_title(f"{title} — first {len(means)} steps")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return fig