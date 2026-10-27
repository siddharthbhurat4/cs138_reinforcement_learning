"""Train the Part 2 learner (car 2, with car 1 on the track) with several discount factors on
TRACK_1 and TRACK_2 and plot the learning curves.

    python -m homework2.gamma_sweep_dual               # train all runs, then plot
    python -m homework2.gamma_sweep_dual --plot-only   # re-plot from the saved episode lengths

One figure per track, with one learning curve (episode length of car 2, smoothed) per gamma.
Car 1 uses the Part 1 agent trained on the same track (AGENT1_FILES below).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# EDIT if your Part 2 code lives in a different module
from homework2.main_dual_agent import Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, make_grid
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

GAMMAS = [0.3, 0.6, 1.0]
TRACKS = {"track1": TRACK_1, "track2": TRACK_2}
AGENT1_FILES = {"track1": "homework2/results/hw2_mc_agent1_track1.pkl",
                "track2": "homework2/results/hw2_mc_agent1_track2.pkl"}
NUM_EPISODES = 10000
NUM_STEPS = 1000            # per-episode step limit
RELATIVE_RANGE = 3
EPSILON = 0.1
SEED = 0
WINDOW = 200                # moving-average window for the curves
RESULTS_DIR = "homework2/results/gamma_sweep_dual"
PLOT_DIR = "homework2/plot_results"


def lengths_file(track, gamma):
    return os.path.join(RESULTS_DIR, f"lengths_{track}_gamma{gamma}.npy")


def train(track, gamma):
    """Train one car-2 agent from scratch and save its episode lengths."""
    set_global_seed(SEED)
    grid = make_grid(TRACKS[track])
    size = 2 * RELATIVE_RANGE + 1
    env = RaceTrack(grid=grid, grid_size=grid.shape[1],
                    observation_space=(grid.shape[1], grid.shape[0], 5, 5, size, size),
                    max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, relative_range=RELATIVE_RANGE, seed=SEED)
    env.load(AGENT1_FILES[track])
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
    ax.set_ylabel("Episode length of car 2 (steps)")
    ax.set_title(f"{track}, two cars: learning curves for different γ (moving average over {WINDOW} episodes)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, f"hw2_gamma_sweep_dual_{track}.png")
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