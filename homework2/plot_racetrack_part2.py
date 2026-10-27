"""Visualise Part 2 (two cars): car 1 follows the Part 1 policy, car 2 is the learner.

Run after the Part 2 training script has saved its agent and episode lengths:

    python -m homework2.plot_racetrack_part2 --track track1                    # show the figures
    python -m homework2.plot_racetrack_part2 --track track2 --episodes 4 --save   # save them instead
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# EDIT if your Part 2 code lives in a different module
from homework2.main_dual_agent import FINISH, OFF_TRACK, START, TRACK, Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, make_grid
from rl.utils import moving_average

# File names per track; {name} is replaced by the track's file name below
AGENT1_FILE = "homework2/results/hw2_mc_agent1_{name}.pkl"                 # Part 1 policy (car 1)
AGENT2_FILE = "homework2/results/hw2_mc_part2_{name}.pkl"                  # Part 2 learner (car 2)
LENGTHS_FILE = "homework2/results/hw2_episode_lengths_part2_{name}.npy"
PART1_LENGTHS_FILE = "homework2/results/hw2_episode_lengths_agent1_{name}.npy"   # optional, for comparison
FILE_NAMES = {"small": "mygrid", "track1": "track1", "track2": "track2"}
PLOT_DIR = "homework2/plot_results"
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}
CAR1_COLOR, CAR2_COLOR = "#d95f02", "#1b6ac9"


# ------------------------------------------------------------------ environment setup
def small_grid(size=8, block=5):
    """The 8x8 right-turn track, built exactly as in main()."""
    grid = np.full((size, size), TRACK, dtype=int)
    grid[:block, size - block:] = OFF_TRACK
    grid[0, :size - block] = START
    grid[block:, size - 1] = FINISH
    return grid


GRIDS = {"small": small_grid, "track1": lambda: make_grid(TRACK_1), "track2": lambda: make_grid(TRACK_2)}


def make_env(grid, agent1_file, max_steps, relative_range, seed):
    """Build the Part 2 environment exactly as main() does - EDIT if you change main()."""
    h, w = grid.shape
    size = 2 * relative_range + 1
    env = RaceTrack(grid=grid, grid_size=w, observation_space=(w, h, 5, 5, size, size),
                    max_episodes=max_steps, epsilon_vel_noise=0.1, relative_range=relative_range, seed=seed)
    env.load(agent1_file)
    return env


def track_restarts(env):
    """Count restarts (crash or collision) per car by watching env._start_cell_avoiding.

    Car 2 restarts avoiding car 1's cell, car 1 restarts avoiding car 2's cell, so the
    arguments tell us which car restarted. The environment itself is not changed.
    """
    original = env._start_cell_avoiding
    counter = {"car1": 0, "car2": 0}

    def wrapped(x_avoid, y_avoid):
        counter["car2" if (x_avoid, y_avoid) == (env.x1, env.y1) else "car1"] += 1
        return original(x_avoid, y_avoid)

    env._start_cell_avoiding = wrapped
    return counter


# ------------------------------------------------------------------ running episodes
def _positions(env):
    """Both cars' true positions, read from the environment (the observation only has car 1 relative to car 2)."""
    return {"car1": (int(env.x1), int(env.y1)), "car2": (int(env.x2), int(env.y2))}


def greedy_episode(env, agent, counter):
    """One greedy episode. Returns per-step records and a summary dict."""
    agent.eval()
    obs, _ = env.reset()
    steps = [{**_positions(env), "restart1": False, "restart2": False}]
    car1_finish_step = None
    while True:
        before = dict(counter)
        result = env.step(agent.act(obs))
        obs = result.observation
        steps.append({**_positions(env),
                      "restart1": counter["car1"] > before["car1"],
                      "restart2": counter["car2"] > before["car2"]})
        if car1_finish_step is None and env.agent1_at_goal:
            car1_finish_step = len(steps) - 1
        if result.done:
            break
    summary = {"length": len(steps) - 1, "finished": result.terminated,
               "car2_restarts": sum(s["restart2"] for s in steps),
               "car1_restarts": sum(s["restart1"] for s in steps),
               "car1_finish_step": car1_finish_step}
    return steps, summary


# ------------------------------------------------------------------ drawing helpers
def _is_big(grid):
    return max(grid.shape) > 10


def _panel_size(grid, width=4.4):
    """Panel size that keeps grid cells square, whatever the track's shape."""
    n_rows, n_cols = grid.shape
    return width, max(3.5, width * n_rows / n_cols) + 0.6


def draw_track(ax, grid):
    """Draw the grid with y = 0 at the bottom (origin='lower' matches grid[y, x])."""
    codes = sorted(TRACK_COLORS)
    ax.imshow(grid, origin="lower", cmap=ListedColormap([TRACK_COLORS[c] for c in codes]),
              vmin=min(codes) - 0.5, vmax=max(codes) + 0.5)
    n_rows, n_cols = grid.shape
    tick_step = 5 if _is_big(grid) else 1
    ax.set_xticks(range(0, n_cols, tick_step))
    ax.set_yticks(range(0, n_rows, tick_step))
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6 if _is_big(grid) else 1.5)
    ax.tick_params(which="minor", length=0)


def path_with_breaks(steps, car):
    """x, y lists for one car, with NaN gaps where it restarted (so no line jumps to the start)."""
    xs, ys = [], []
    for k, s in enumerate(steps):
        if k > 0 and s[f"restart{car}"]:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(s[f"car{car}"][0])
        ys.append(s[f"car{car}"][1])
    return xs, ys


def _stays(steps, car):
    """Group consecutive steps where a car is in the same cell: yields ((x, y), first_step, last_step)."""
    groups = []
    for k, st in enumerate(steps):
        pos = st[f"car{car}"]
        if groups and groups[-1][0] == pos and groups[-1][2] == k - 1:
            groups[-1][2] = k
        else:
            groups.append([pos, k, k])
    return groups


def draw_episode(ax, grid, steps, summary):
    draw_track(ax, grid)
    big = _is_big(grid)
    for car, color, style in ((1, CAR1_COLOR, "--"), (2, CAR2_COLOR, "-")):
        xs, ys = path_with_breaks(steps, car)
        ax.plot(xs, ys, style, color=color, linewidth=1.4 if big else 1.8, marker="o",
                markersize=2.5 if big else 3.5, alpha=0.9)
        ax.plot(xs[0], ys[0], "s", color=color, markersize=6 if big else 8)
        # step numbers next to each position; a car that stays put gets one label like "3-90"
        dx, dy = (-0.3, 0.2) if car == 1 else (0.14, -0.3)
        for pos, first, last in _stays(steps, car):
            label = str(first) if first == last else f"{first}-{last}"
            ax.text(pos[0] + dx, pos[1] + dy, label, color=color, fontsize=5 if big else 6.5,
                    ha="center", va="center")
    car1 = (f"car 1 finished at step {summary['car1_finish_step']}" if summary["car1_finish_step"]
            else "car 1 did not finish")
    car2 = f"car 2: {summary['length']} steps" + ("" if summary["finished"] else " (not finished)")
    ax.set_title(f"{car2}, {summary['car2_restarts']} restart(s)\n{car1}", fontsize=9)


# ------------------------------------------------------------------ figures
def plot_episodes(env, agent, counter, n_episodes, save_path=None):
    n_cols = min(n_episodes, 3)
    n_rows = -(-n_episodes // n_cols)
    panel_w, panel_h = _panel_size(env.grid)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, panel_h * n_rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i >= n_episodes:
            ax.axis("off")
            continue
        steps, summary = greedy_episode(env, agent, counter)
        draw_episode(ax, env.grid, steps, summary)
    handles = [plt.Line2D([], [], color=CAR1_COLOR, linestyle="--", marker="o", label="car 1 (Part 1 policy)"),
               plt.Line2D([], [], color=CAR2_COLOR, linestyle="-", marker="o", label="car 2 (learner, greedy)"),
               plt.Line2D([], [], color="grey", linestyle="", marker="s", label="start position")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Greedy episodes with two cars (numbers = step at which each car was there)", y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.subplots_adjust(hspace=0.3 if _is_big(env.grid) else 0.42)
    _finish(fig, save_path)


def plot_learning_curves(lengths_file, part1_lengths_file, window=200, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    curves = {"Part 2: car 2 with car 1 on the track": lengths_file,
              "Part 1: single car": part1_lengths_file}
    plotted = False
    for (label, path), color in zip(curves.items(), (CAR2_COLOR, "grey")):
        if not os.path.exists(path):
            print(f"No {path} found - skipping that curve.")
            continue
        lengths = np.load(path)
        smoothed = moving_average(lengths, window)
        ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, color=color, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"Learning curves (moving average over {window} episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path)


def plot_exploration(env, agent, save_path=None):
    """Updates per cell for car 2, summed over everything else in its observation and the actions."""
    counts = agent.state_action_taken_count           # axes: x2, y2, vx2, vy2, rel_x, rel_y, action
    visits = counts.sum(axis=tuple(range(2, counts.ndim)))   # -> [x2, y2]
    panel_w, panel_h = _panel_size(env.grid, width=5.5)
    fig, ax = plt.subplots(figsize=(panel_w + 1, panel_h))
    shown = np.where(env.grid == OFF_TRACK, np.nan, visits.T)
    im = ax.imshow(np.log10(shown + 1), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="log10(updates + 1)")
    tick_step = 5 if _is_big(env.grid) else 1
    ax.set_xticks(range(0, env.grid.shape[1], tick_step))
    ax.set_yticks(range(0, env.grid.shape[0], tick_step))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("How often car 2 was learned from in each cell")
    fig.tight_layout()
    _finish(fig, save_path)


def print_summary(env, agent, counter, n):
    results = [greedy_episode(env, agent, counter)[1] for _ in range(n)]
    finished = [r for r in results if r["finished"]]
    lengths = np.array([r["length"] for r in finished]) if finished else np.array([np.nan])
    car2_first = sum(1 for r in finished if r["car1_finish_step"] is None or r["length"] < r["car1_finish_step"])
    print(f"\nGreedy summary over {n} episodes (velocity noise on):")
    print(f"  car 2 finished:                 {len(finished)}/{n}")
    print(f"  car 2 lap length (finished):    mean {np.nanmean(lengths):.2f}, best {np.nanmin(lengths):.0f}, worst {np.nanmax(lengths):.0f}")
    print(f"  car 2 restarts (crash or collision) per episode: {np.mean([r['car2_restarts'] for r in results]):.2f}")
    print(f"  car 2 finished before car 1:    {car2_first}/{n}")


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
    parser.add_argument("--track", default="track1", choices=sorted(GRIDS), help="the track both agents were trained on")
    parser.add_argument("--episodes", type=int, default=5, help="greedy episodes to draw")
    parser.add_argument("--summary", type=int, default=200, help="greedy episodes for the printed summary")
    parser.add_argument("--max-steps", type=int, default=1000, help="per-episode step limit (as in main())")
    parser.add_argument("--relative-range", type=int, default=3, help="RELATIVE_RANGE used in main()")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", action="store_true", help=f"save figures to {PLOT_DIR}/ instead of showing them")
    args = parser.parse_args()

    name = FILE_NAMES[args.track]
    grid = GRIDS[args.track]()
    env = make_env(grid, AGENT1_FILE.format(name=name), args.max_steps, args.relative_range, args.seed)
    counter = track_restarts(env)
    agent = Jumbo(env.observation_space, env.action_space, seed=args.seed)
    agent.load(AGENT2_FILE.format(name=name))
    if agent.state_action_taken_count.shape[:-1] != tuple(env.observation_space.sizes):
        raise SystemExit(f"{AGENT2_FILE.format(name=name)} does not match --track {args.track} / "
                         f"--relative-range {args.relative_range} (Q-table shape differs)")

    def out(fig_name):
        return os.path.join(PLOT_DIR, f"hw2_part2_{args.track}_{fig_name}.png") if args.save else None

    plot_episodes(env, agent, counter, args.episodes, out("episodes"))
    plot_learning_curves(LENGTHS_FILE.format(name=name), PART1_LENGTHS_FILE.format(name=name),
                         save_path=out("learning_curves"))
    plot_exploration(env, agent, out("exploration"))
    print_summary(env, agent, counter, args.summary)


if __name__ == "__main__":
    main()