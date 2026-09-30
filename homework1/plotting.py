"""Plotting helpers for Homework 1 (multi-armed bandits).

Usage from homework1/main.py:

    from homework1.plotting import plot_bandit_results

    results = {
        "sample average": (rewards, is_optimal),        # each of shape (runs, steps)
        "constant step size": (rewards2, is_optimal2),
    }
    plot_bandit_results(results, layout="overlay")   # all methods on the same axes
    plot_bandit_results(results, layout="columns")   # one column per method, side by side
    plot_bandit_results(results, save_path="results/hw1.png")   # save for the report
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from rl.utils import moving_average


def _per_step_average(data):
    """Average over runs: accepts shape (runs, steps) or (steps,), returns shape (steps,)."""
    return np.atleast_2d(np.asarray(data, dtype=float)).mean(axis=0)


def _smoothed_with_steps(values, window):
    """Smooth with a moving average and return the step number each smoothed point ends at."""
    smoothed = moving_average(values, window)
    first_step = len(values) - len(smoothed) + 1   # steps are numbered from 1
    return np.arange(first_step, len(values) + 1), smoothed


def plot_bandit_results(results, window=100, title="Non-stationary 10-armed bandit",
                        layout="overlay", save_path=None):
    """Plot average reward and % optimal action per step for one or more methods.

    results:   {label: (rewards, is_optimal)}, each a list/array of shape (runs, steps) or (steps,).
    window:    moving-average window in steps (1 = no smoothing).
    layout:    "overlay" - all methods as lines on the same two charts (best for direct comparison)
               "columns" - one column per method, side by side, with shared y-axes per row
    save_path: if given, save the figure there instead of showing it.
    """
    n = len(results)
    if layout == "overlay":
        fig, (ax_r, ax_o) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes_per_method = [(ax_r, ax_o)] * n
        all_axes = [ax_r, ax_o]
    elif layout == "columns":
        fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7), sharex=True, sharey="row", squeeze=False)
        axes_per_method = [(axes[0, i], axes[1, i]) for i in range(n)]
        all_axes = list(axes.flat)
    else:
        raise ValueError(f"layout must be 'overlay' or 'columns', got {layout!r}")

    for i, (label, (rewards, is_optimal)) in enumerate(results.items()):
        ax_reward, ax_optimal = axes_per_method[i]
        color = f"C{i}"                                   # same colour per method in both layouts
        n_runs = np.atleast_2d(np.asarray(rewards)).shape[0]
        full_label = label + (f" ({n_runs} runs)" if n_runs > 1 else "")

        steps, avg_reward = _smoothed_with_steps(_per_step_average(rewards), window)
        ax_reward.plot(steps, avg_reward, color=color, label=full_label)

        steps, pct_optimal = _smoothed_with_steps(100 * _per_step_average(is_optimal), window)
        ax_optimal.plot(steps, pct_optimal, color=color, label=full_label)

        if layout == "columns":
            ax_reward.set_title(full_label)

    # labels: y on the left column, x on the bottom row
    first_reward, first_optimal = axes_per_method[0]
    first_reward.set_ylabel("Average reward")
    first_optimal.set_ylabel("% optimal action")
    for _, ax_optimal in axes_per_method:
        ax_optimal.set_xlabel("Step")
        ax_optimal.set_ylim(0, 100)
    for ax in all_axes:
        ax.grid(alpha=0.3)
    if layout == "overlay":
        for ax in all_axes:
            ax.legend()

    smoothing_note = f" — moving average over {window} steps" if window > 1 else ""
    fig.suptitle(title + smoothing_note)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig