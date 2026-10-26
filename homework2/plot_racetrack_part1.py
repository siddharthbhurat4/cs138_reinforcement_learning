"""Visualise a trained racetrack agent (run after homework2.main has saved it).

    python -m homework2.plot_racetrack --track track1                # show the figures
    python -m homework2.plot_racetrack --track track2 --routes 5 --save   # save them instead

    --track: small (the 8x8 track), track1 or track2 (the book's tracks) - use the one the agent was trained on

Figure 1: the track with several greedy routes from the start line (noise on).
Figure 2: how often each cell was explored during training.
Figure 3: the learning curve, if the episode-lengths file exists.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from homework2.main import FINISH, OFF_TRACK, START, TRACK, Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, TRACK_3, make_grid
from rl.utils import moving_average

AGENT_FILE = "homework2/results/hw2_mc_agent1_track2.pkl"
LENGTHS_FILE = "homework2/results/hw2_episode_lengths_agent1_track2.npy"
PLOT_DIR = "homework2/plot_results"
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}


def small_grid(size=8, block=5):
    """The 8x8 right-turn track (bottom-right block x block corner off-track)."""
    grid = np.full((size, size), TRACK, dtype=int)
    grid[:block, size - block:] = OFF_TRACK
    grid[0, :size - block] = START
    grid[block:, size - 1] = FINISH
    return grid


GRIDS = {"small": small_grid, "track1": lambda: make_grid(TRACK_1), "track2": lambda: make_grid(TRACK_2), "track3": lambda: make_grid(TRACK_3)}


def make_env(grid, seed):
    """Build the environment exactly as main.py does."""
    height, width = grid.shape
    return RaceTrack(grid_size=0, observation_space=(width, height, 5, 5), grid=grid,
                     max_episodes=1000, epsilon_vel_noise=0.1, seed=seed)


def _figsize(grid, width=6.0):
    """Figure size that keeps grid cells square, whatever the track's shape."""
    n_rows, n_cols = grid.shape
    return width, max(3.0, width * n_rows / n_cols)


def greedy_route(env, agent):
    """One greedy episode from the start line. Returns the list of (x, y, vx, vy) and whether it finished."""
    agent.eval()
    obs, _ = env.reset()
    route = [obs]
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        route.append(obs)
        if result.done:
            return route, result.terminated


def draw_track(ax, grid):
    """Draw the grid with y = 0 at the bottom (origin='lower' matches grid[y, x])."""
    codes = sorted(TRACK_COLORS)
    ax.imshow(grid, origin="lower", cmap=ListedColormap([TRACK_COLORS[c] for c in codes]),
              vmin=min(codes) - 0.5, vmax=max(codes) + 0.5)
    n_rows, n_cols = grid.shape
    tick_step = 1 if max(n_rows, n_cols) <= 10 else 5           # fewer labels on big tracks
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_yticks(range(0, n_rows, tick_step))
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5 if max(n_rows, n_cols) <= 10 else 0.6)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_routes(env, agent, n_routes, save_path=None):
    fig, ax = plt.subplots(figsize=_figsize(env.grid))
    draw_track(ax, env.grid)
    for i in range(n_routes):
        route, finished = greedy_route(env, agent)
        xs = [s[0] for s in route]
        ys = [s[1] for s in route]
        # a crash teleports the car back to the start line: break the line there instead of drawing a jump
        for k in range(1, len(route)):
            if route[k][2] == 0 and route[k][3] == 0 and env.grid[route[k][1], route[k][0]] == START \
                    and env.grid[route[k - 1][1], route[k - 1][0]] != START:
                xs[k] = ys[k] = np.nan
        color = f"C{i % 10}"
        ax.plot(xs, ys, "-o", color=color, markersize=3, linewidth=1.5, alpha=0.85,
                label=f"route {i + 1}: {len(route) - 1} steps" + ("" if finished else " (not finished)"))
        ax.plot(xs[0], ys[0], "s", color=color, markersize=7)    # start marker
    ax.set_title("Greedy routes from the start line")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    _finish(fig, save_path)


def plot_exploration(env, agent, save_path=None):
    """Total number of first-visit updates per cell, summed over all velocities and actions."""
    visits = agent.state_action_taken_count.sum(axis=(2, 3, 4))   # indexed [x, y]
    width, height = _figsize(env.grid)
    fig, ax = plt.subplots(figsize=(width + 1, height))
    shown = np.where(env.grid == OFF_TRACK, np.nan, visits.T)     # .T -> [y, x] like the grid
    im = ax.imshow(np.log10(shown + 1), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10(updates + 1)")
    n_rows, n_cols = env.grid.shape
    tick_step = 1 if max(n_rows, n_cols) <= 10 else 5
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_yticks(range(0, n_rows, tick_step))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("How often each cell was learned from (off-track cells blank)")
    fig.tight_layout()
    _finish(fig, save_path)


def plot_learning_curve(lengths, window=200, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    smoothed = moving_average(lengths, window)
    ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"Learning curve (moving average over {window} episodes)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, save_path)


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="track1", choices=sorted(GRIDS), help="the track the agent was trained on")
    parser.add_argument("--agent", default=AGENT_FILE)
    parser.add_argument("--lengths", default=LENGTHS_FILE)
    parser.add_argument("--routes", type=int, default=20, help="number of greedy routes to draw")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", action="store_true", help=f"save figures to {PLOT_DIR}/ instead of showing them")
    args = parser.parse_args()

    env = make_env(GRIDS[args.track](), args.seed)
    agent = Jumbo(env.observation_space, env.action_space, seed=args.seed)
    agent.load(args.agent)
    if agent.state_action_taken_count.shape[:2] != (env.grid.shape[1], env.grid.shape[0]):
        raise SystemExit(f"{args.agent} was not trained on --track {args.track} (Q-table size does not match)")

    def out(name):
        return os.path.join(PLOT_DIR, f"hw2_{args.track}_{name}.png") if args.save else None

    plot_routes(env, agent, args.routes, out("routes"))
    plot_exploration(env, agent, out("exploration"))
    if os.path.exists(args.lengths):
        plot_learning_curve(np.load(args.lengths), save_path=out("learning_curve"))
    else:
        print(f"No {args.lengths} found - save the episode lengths in main.py to get the learning curve.")


if __name__ == "__main__":
    main()