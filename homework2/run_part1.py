"""Part 1: single car. Trains on the 8x8 grid, track 1 and track 2 with gamma = 0.3, 0.6 and 1.0,
then shows the learning curves and greedy routes. Nothing is saved; save figures from the plot window.

    python -m homework2.run_part1          # run from the reinforcement_learning/ folder
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from homework2.main import FINISH, OFF_TRACK, START, TRACK, Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, make_grid
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

GAMMAS = [0.3, 0.6, 1.0]
EPSILON = 0.1
SEED = 0
WINDOW = 200            # moving-average window for the learning curves
N_ROUTES = 3            # greedy routes drawn per track and gamma
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}


def small_grid(size=8, block=5):
    """The 8x8 right-turn track (bottom-right block x block corner off-track)."""
    grid = np.full((size, size), TRACK, dtype=int)
    grid[:block, size - block:] = OFF_TRACK
    grid[0, :size - block] = START
    grid[block:, size - 1] = FINISH
    return grid


# track name -> (grid, number of training episodes, per-episode step limit)
TRACKS = {
    "mygrid": (small_grid(), 5000, 200),
    "track1": (make_grid(TRACK_1), 10000, 1000),
    "track2": (make_grid(TRACK_2), 10000, 1000),
}


def make_env(grid, max_steps, seed=SEED):
    h, w = grid.shape
    return RaceTrack(grid=grid, grid_size=w, observation_space=(w, h, 5, 5),
                     max_episodes=max_steps, epsilon_vel_noise=0.1, seed=seed)


def greedy_route(env, agent):
    """One greedy episode from the start line: x and y lists (NaN gaps at restarts), steps, finished."""
    agent.eval()
    obs, _ = env.reset()
    xs, ys = [obs[0]], [obs[1]]
    steps = 0
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        steps += 1
        if (obs[2] == 0 and obs[3] == 0 and env.grid[obs[1], obs[0]] == START
                and not np.isnan(xs[-1]) and env.grid[ys[-1], xs[-1]] != START):
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(obs[0])
        ys.append(obs[1])
        if result.done:
            return xs, ys, steps, result.terminated


def train(track, gamma):
    """Train one agent; return its episode lengths and a few greedy routes (the agent is then discarded)."""
    grid, num_episodes, max_steps = TRACKS[track]
    set_global_seed(SEED)
    env = make_env(grid, max_steps)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, gamma=gamma, seed=SEED)
    print(f"\n=== {track}, gamma = {gamma} ===")
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=max(num_episodes // 5, 1))])
    history = runner.train(num_episodes)
    print("Evaluation (greedy, 100 episodes):", runner.evaluate(num_episodes=100))
    route_env = make_env(grid, max_steps, seed=SEED + 1)
    routes = [greedy_route(route_env, agent) for _ in range(N_ROUTES)]
    return [h.length for h in history], routes


def draw_track(ax, grid):
    codes = sorted(TRACK_COLORS)
    ax.imshow(grid, origin="lower", cmap=ListedColormap([TRACK_COLORS[c] for c in codes]),
              vmin=min(codes) - 0.5, vmax=max(codes) + 0.5)
    n_rows, n_cols = grid.shape
    step = 5 if max(n_rows, n_cols) > 10 else 1
    ax.set_xticks(range(0, n_cols, step))
    ax.set_yticks(range(0, n_rows, step))
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)


def plot_learning_curves(results):
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(5.5 * len(TRACKS), 4.5), constrained_layout=True)
    for ax, track in zip(axes, TRACKS):
        for gamma in GAMMAS:
            lengths = results[track][gamma][0]
            smoothed = moving_average(lengths, WINDOW)
            ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, label=f"γ = {gamma}")
        ax.set_title(track)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode length (steps)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Part 1: learning curves (moving average over {WINDOW} episodes)")


def plot_routes(results):
    """One figure per track (a row of panels, one per gamma), sized to fit on a normal screen."""
    for track in TRACKS:
        grid = TRACKS[track][0]
        n_rows, n_cols = grid.shape
        panel_h = min(7.0, 4.0 * n_rows / n_cols)          # tall tracks get taller panels, capped for the screen
        fig, axes = plt.subplots(1, len(GAMMAS), figsize=(4.4 * len(GAMMAS), panel_h + 2.4),
                                 constrained_layout=True)
        for ax, gamma in zip(axes, GAMMAS):
            draw_track(ax, grid)
            for i, (xs, ys, steps, finished) in enumerate(results[track][gamma][1]):
                ax.plot(xs, ys, "-o", color=f"C{i}", markersize=3, linewidth=1.5, alpha=0.85,
                        label=f"route {i + 1}: {steps} steps" + ("" if finished else " (not finished)"))
            ax.set_title(f"γ = {gamma}", fontsize=11)
            # legend below the panel, outside the plotting area, so it never covers the track
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False)
        fig.suptitle(f"Part 1: greedy routes on {track} (velocity noise on)", fontsize=13)


def main():
    results = {track: {gamma: train(track, gamma) for gamma in GAMMAS} for track in TRACKS}
    plot_learning_curves(results)
    plot_routes(results)
    plt.show()


if __name__ == "__main__":
    main()