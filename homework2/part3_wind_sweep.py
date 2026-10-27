"""Train the Part 1 agent on the windy STAIRCASE with several wind probabilities, then plot
greedy routes (one panel per wind probability) and the learning curves (one plot, one curve each).

    python -m homework2.part3_wind_sweep               # train all runs, then plot
    python -m homework2.part3_wind_sweep --plot-only   # re-plot from the saved agents and episode lengths
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# EDIT if your Part 1 code lives in a different module
from homework2.main import FINISH, OFF_TRACK, START, TRACK, Jumbo
from homework2.racetracks import TRACK_3, make_grid
from homework2.windy import WindyRaceTrack, staircase_wind
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

WIND_PROBS = [0.25, 0.5, 1.0]
NUM_EPISODES = 10000
NUM_STEPS = 2000            # per-episode step limit
EPSILON = 0.1
GAMMA = 1
SEED = 0
WINDOW = 200                # moving-average window for the learning curves
N_ROUTES = 3                # greedy routes drawn per wind probability
RESULTS_DIR = "homework2/results/wind_sweep"
PLOT_DIR = "homework2/plot_results"
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}


def agent_file(p):
    return os.path.join(RESULTS_DIR, f"agent_staircase_wind{p}.pkl")


def lengths_file(p):
    return os.path.join(RESULTS_DIR, f"lengths_staircase_wind{p}.npy")


def make_env(wind_prob, seed=SEED, max_steps=NUM_STEPS):
    grid = make_grid(TRACK_3)
    return WindyRaceTrack(grid=grid, grid_size=grid.shape[1], observation_space=(grid.shape[1], grid.shape[0], 5, 5),
                          max_episodes=max_steps, epsilon_vel_noise=0.1, seed=seed,
                          wind_x=staircase_wind(grid), wind_prob=wind_prob)


def train(wind_prob):
    """Train one agent from scratch, save it and its episode lengths, and print a greedy evaluation."""
    set_global_seed(SEED)
    env = make_env(wind_prob)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, gamma=GAMMA, seed=SEED)
    print(f"\n=== wind_prob = {wind_prob} ===")
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=max(NUM_EPISODES // 5, 1))])
    history = runner.train(NUM_EPISODES)
    print("Evaluation (greedy, 100 episodes):", runner.evaluate(num_episodes=100))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent.save(agent_file(wind_prob))
    np.save(lengths_file(wind_prob), [h.length for h in history])


# ------------------------------------------------------------------ plotting
def greedy_route(env, agent):
    """One greedy episode from the start line: x and y lists (NaN gaps at restarts) and whether it finished."""
    agent.eval()
    obs, _ = env.reset()
    xs, ys = [obs[0]], [obs[1]]
    steps = 0
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        steps += 1
        restarted = (obs[2] == 0 and obs[3] == 0 and env.grid[obs[1], obs[0]] == START
                     and not np.isnan(xs[-1]) and env.grid[ys[-1], xs[-1]] != START)
        if restarted:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(obs[0])
        ys.append(obs[1])
        if result.done:
            return xs, ys, steps, result.terminated


def draw_track(ax, grid, wind):
    codes = sorted(TRACK_COLORS)
    ax.imshow(grid, origin="lower", cmap=ListedColormap([TRACK_COLORS[c] for c in codes]),
              vmin=min(codes) - 0.5, vmax=max(codes) + 0.5)
    wy, wx = np.nonzero(wind)
    ax.scatter(wx, wy, marker=r"$\rightarrow$", s=45, color="#7a9cc6", alpha=0.6)
    n_rows, n_cols = grid.shape
    ax.set_xticks(range(0, n_cols, 5))
    ax.set_yticks(range(0, n_rows, 5))
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)


def plot_routes(save=True):
    fig, axes = plt.subplots(1, len(WIND_PROBS), figsize=(5.2 * len(WIND_PROBS), 5.6), squeeze=False)
    for ax, p in zip(axes[0], WIND_PROBS):
        if not os.path.exists(agent_file(p)):
            print(f"No {agent_file(p)} - skipping wind_prob = {p}")
            ax.axis("off")
            continue
        env = make_env(p, seed=SEED + 1, max_steps=300)
        agent = Jumbo(env.observation_space, env.action_space, seed=SEED + 1)
        agent.load(agent_file(p))
        draw_track(ax, env.grid, env.wind_x)
        for i in range(N_ROUTES):
            xs, ys, steps, finished = greedy_route(env, agent)
            ax.plot(xs, ys, "-o", color=f"C{i}", markersize=3, linewidth=1.6, alpha=0.85,
                    label=f"route {i + 1}: {steps} steps" + ("" if finished else " (not finished)"))
        ax.set_title(f"wind_prob = {p}")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Greedy routes on the windy staircase (wind → pushes right; velocity noise on)")
    fig.tight_layout()
    _save(fig, "hw2_wind_sweep_routes.png", save)


def plot_learning_curves(save=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in WIND_PROBS:
        if not os.path.exists(lengths_file(p)):
            print(f"No {lengths_file(p)} - skipping wind_prob = {p}")
            continue
        lengths = np.load(lengths_file(p))
        smoothed = moving_average(lengths, WINDOW)
        ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, label=f"wind_prob = {p}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"Windy staircase: learning curves (moving average over {WINDOW} episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, "hw2_wind_sweep_learning_curves.png", save)


def _save(fig, name, save):
    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, name)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true", help="skip training, plot from the saved results")
    args = parser.parse_args()

    if not args.plot_only:
        for p in WIND_PROBS:
            train(p)
    plot_routes()
    plot_learning_curves()
    plt.show()


if __name__ == "__main__":
    main()