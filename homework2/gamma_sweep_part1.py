"""Train the Part 1 agent with several discount factors on TRACK_1 and TRACK_2 and plot the learning curves.

    python -m homework2.gamma_sweep               # train all runs, then plot
    python -m homework2.gamma_sweep --plot-only   # re-plot from the saved episode lengths

One figure per track, with one learning curve (episode length, smoothed) per gamma.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# EDIT if your Part 1 code lives in a different module
from homework2.main import Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, make_grid
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

GAMMAS = [1.0, 0.6, 0.3]
TRACKS = {"track1": TRACK_1, "track2": TRACK_2}
NUM_EPISODES = 5000
NUM_STEPS = 1000            # per-episode step limit
EPSILON = 0.1
SEED = 0
WINDOW = 200                # moving-average window for the curves
RESULTS_DIR = "homework2/results/gamma_sweep"
PLOT_DIR = "homework2/plot_results"


def lengths_file(track, gamma):
    return os.path.join(RESULTS_DIR, f"lengths_{track}_gamma{gamma}.npy")


def train(track, gamma):
    """Train one agent from scratch and save its episode lengths."""
    set_global_seed(SEED)
    grid = make_grid(TRACKS[track])
    env = RaceTrack(grid=grid, grid_size=grid.shape[1], observation_space=(grid.shape[1], grid.shape[0], 5, 5),
                    max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, seed=SEED)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, gamma=gamma, seed=SEED)
    print(f"\n=== {track}, gamma = {gamma} ===")
    history = Runner(env, agent, callbacks=[ProgressPrinter(every=max(NUM_EPISODES // 5, 1))]).train(NUM_EPISODES)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(lengths_file(track, gamma), [h.length for h in history])


def plot(track, save=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for gamma in GAMMAS:
        path = lengths_file(track, gamma)
        if not os.path.exists(path):
            print(f"No {path} - skipping gamma = {gamma}")
            continue
        lengths = np.load(path)
        smoothed = moving_average(lengths, WINDOW)
        ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, label=f"γ = {gamma}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"{track}: learning curves for different γ (moving average over {WINDOW} episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, f"hw2_gamma_sweep_{track}.png")
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="skip training, plot the saved lengths")
    args = parser.parse_args()

    if not args.plot_only:
        for track in TRACKS:
            for gamma in GAMMAS:
                train(track, gamma)
    for track in TRACKS:
        plot(track)
    plt.show()


if __name__ == "__main__":
    main()