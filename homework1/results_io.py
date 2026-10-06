"""Save experiment results to disk and load them back, so plots can be redrawn
without re-running training.

    from homework1.results_io import save_results, load_results

    save_results(results, "results/hw1_bandit.npz", settings={"epsilon": 0.1, ...})
    results, settings = load_results("results/hw1_bandit.npz")

The file is a compressed NumPy archive (.npz). results/ and *.npz are in .gitignore.
"""
import json
import os

import numpy as np


def save_results(results, path, settings=None, full=False):
    """Save {label: (rewards, is_optimal)} where each array has shape (runs, steps).

    By default only per-step statistics across runs are saved (mean and std of the
    reward, fraction of optimal actions) - all the plots need, and a tiny file.
    full=True also saves every run's raw data (large: ~40 MB per method for 1000 x 10000).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {
        "labels": np.array(list(results.keys())),
        "settings": np.array(json.dumps(settings or {})),
    }
    for i, (rewards, is_optimal) in enumerate(results.values()):
        rewards = np.atleast_2d(np.asarray(rewards, dtype=np.float32))
        is_optimal = np.atleast_2d(np.asarray(is_optimal, dtype=bool))
        arrays[f"m{i}_n_runs"] = np.array(rewards.shape[0])
        arrays[f"m{i}_reward_mean"] = rewards.mean(axis=0)
        arrays[f"m{i}_reward_std"] = rewards.std(axis=0)
        arrays[f"m{i}_optimal_fraction"] = is_optimal.mean(axis=0)
        if full:
            arrays[f"m{i}_rewards"] = rewards
            arrays[f"m{i}_is_optimal"] = is_optimal

    np.savez_compressed(path, **arrays)
    print(f"Saved results to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_results(path, raw=False):
    """Load what save_results wrote.

    Returns (results, settings). results is {label: (rewards, is_optimal)} ready for
    plot_bandit_results. With raw=True (and a file saved with full=True) you get the
    per-run arrays of shape (runs, steps); otherwise the per-step averages, with the
    run count added to each label.
    """
    data = np.load(path, allow_pickle=False)
    settings = json.loads(str(data["settings"]))
    results = {}
    for i, label in enumerate(data["labels"]):
        label = str(label)
        if raw:
            if f"m{i}_rewards" not in data.files:
                raise ValueError(f"{path} has no raw data - save it with full=True")
            results[label] = (data[f"m{i}_rewards"], data[f"m{i}_is_optimal"])
        else:
            n_runs = int(data[f"m{i}_n_runs"])
            results[f"{label} ({n_runs} runs)"] = (data[f"m{i}_reward_mean"], data[f"m{i}_optimal_fraction"])
    return results, settings


def load_reward_std(path):
    """Per-step reward standard deviation across runs, {label: array}, e.g. for shaded bands."""
    data = np.load(path, allow_pickle=False)
    return {str(label): data[f"m{i}_reward_std"] for i, label in enumerate(data["labels"])}