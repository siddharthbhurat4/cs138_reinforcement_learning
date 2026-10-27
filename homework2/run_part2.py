"""Part 2: two cars. For track 1 and track 2, first trains car 1 with the Part 1 method (gamma = 1),
then trains car 2 next to it with gamma = 0.3, 0.6 and 0.9, and shows the learning curves and
greedy episodes of both cars. Nothing is saved; save figures from the plot window.

    python -m homework2.run_part2          # run from the reinforcement_learning/ folder
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from homework2.main import Jumbo as SingleCarAgent
from homework2.main import RaceTrack as SingleCarTrack
from homework2.main_dual_agent import FINISH, OFF_TRACK, START, TRACK, Jumbo, RaceTrack
from homework2.racetracks import TRACK_1, TRACK_2, make_grid
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

GAMMAS = [0.3, 0.6, 0.9]
TRACKS = {"track1": make_grid(TRACK_1), "track2": make_grid(TRACK_2)}
CAR1_EPISODES = 10000   # Part 1 training of car 1
CAR2_EPISODES = 15000   # Part 2 training of car 2
NUM_STEPS = 1000        # per-episode step limit
RELATIVE_RANGE = 3
EPSILON = 0.1
SEED = 0
WINDOW = 200            # moving-average window for the learning curves
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}
CAR1_COLOR, CAR2_COLOR = "#d95f02", "#1b6ac9"


def train_car1(track):
    """Train car 1 alone with the Part 1 method; return its Q-table."""
    grid = TRACKS[track]
    h, w = grid.shape
    set_global_seed(SEED)
    env = SingleCarTrack(grid=grid, grid_size=w, observation_space=(w, h, 5, 5),
                         max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, seed=SEED)
    agent = SingleCarAgent(env.observation_space, env.action_space, epsilon=EPSILON, gamma=1, seed=SEED)
    print(f"\n=== {track}: car 1 (Part 1, gamma = 1) ===")
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=max(CAR1_EPISODES // 5, 1))])
    runner.train(CAR1_EPISODES)
    print("Car 1 evaluation (greedy, 100 episodes):", runner.evaluate(num_episodes=100))
    return agent.q_state_action_value


def make_env(track, car1_q, seed=SEED):
    grid = TRACKS[track]
    h, w = grid.shape
    size = 2 * RELATIVE_RANGE + 1
    env = RaceTrack(grid=grid, grid_size=w, observation_space=(w, h, 5, 5, size, size),
                    max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, relative_range=RELATIVE_RANGE, seed=seed)
    env.agent1_q_state_action_value = car1_q      # car 1 drives with the Q-table trained above
    return env


def greedy_episode(env, agent):
    """One greedy episode: both cars' positions per step (read from the environment), steps, finished."""
    agent.eval()
    obs, _ = env.reset()
    car1, car2 = [(env.x1, env.y1)], [(env.x2, env.y2)]
    steps = 0
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        steps += 1
        car1.append((env.x1, env.y1))
        car2.append((env.x2, env.y2))
        if result.done:
            return car1, car2, steps, result.terminated


def train_car2(track, car1_q, gamma):
    """Train car 2 next to car 1; return its episode lengths and one greedy episode (agent then discarded)."""
    set_global_seed(SEED)
    env = make_env(track, car1_q)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, gamma=gamma, seed=SEED)
    print(f"\n=== {track}: car 2, gamma = {gamma} ===")
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=max(CAR2_EPISODES // 5, 1))])
    history = runner.train(CAR2_EPISODES)
    print("Car 2 evaluation (greedy, 100 episodes):", runner.evaluate(num_episodes=100))
    episode = greedy_episode(make_env(track, car1_q, seed=SEED + 1), agent)
    return [h.length for h in history], episode


def _with_breaks(path, grid):
    """x, y lists with a NaN gap wherever the car jumped back to the start line (a restart)."""
    xs, ys = [], []
    for k, (x, y) in enumerate(path):
        if k > 0:
            px, py = path[k - 1]
            if grid[y, x] == START and grid[py, px] != START:
                xs.append(np.nan)
                ys.append(np.nan)
        xs.append(x)
        ys.append(y)
    return xs, ys


def draw_track(ax, grid):
    codes = sorted(TRACK_COLORS)
    ax.imshow(grid, origin="lower", cmap=ListedColormap([TRACK_COLORS[c] for c in codes]),
              vmin=min(codes) - 0.5, vmax=max(codes) + 0.5)
    n_rows, n_cols = grid.shape
    ax.set_xticks(range(0, n_cols, 5))
    ax.set_yticks(range(0, n_rows, 5))
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)


def plot_learning_curves(results):
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(6.5 * len(TRACKS), 4.5), constrained_layout=True)
    for ax, track in zip(axes, TRACKS):
        for gamma in GAMMAS:
            lengths = results[track][gamma][0]
            smoothed = moving_average(lengths, WINDOW)
            ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, label=f"γ = {gamma}")
        ax.set_title(track)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode length of car 2 (steps)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Part 2: learning curves of car 2 (moving average over {WINDOW} episodes)")


def plot_episodes(results):
    """One figure per track (a row of panels, one per gamma), sized to fit on a normal screen."""
    for track in TRACKS:
        grid = TRACKS[track]
        n_rows, n_cols = grid.shape
        panel_h = min(7.0, 4.0 * n_rows / n_cols)          # tall tracks get taller panels, capped for the screen
        fig, axes = plt.subplots(1, len(GAMMAS), figsize=(4.4 * len(GAMMAS), panel_h + 2.4),
                                 constrained_layout=True)
        for ax, gamma in zip(axes, GAMMAS):
            car1, car2, steps, finished = results[track][gamma][1]
            draw_track(ax, grid)
            for path, color, style, label in ((car1, CAR1_COLOR, "--", "car 1 (Part 1 policy)"),
                                              (car2, CAR2_COLOR, "-", "car 2 (learner)")):
                xs, ys = _with_breaks(path, grid)
                ax.plot(xs, ys, style, color=color, marker="o", markersize=2.5, linewidth=1.4, label=label)
                ax.plot(xs[0], ys[0], "s", color=color, markersize=6)
            ax.set_title(f"γ = {gamma}: car 2 took {steps} steps" + ("" if finished else " (not finished)"),
                         fontsize=10)
            # legend below the panel, outside the plotting area, so it never covers the track
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f"Part 2: one greedy episode on {track} (squares = start positions)", fontsize=13)


def main():
    results = {}
    for track in TRACKS:
        car1_q = train_car1(track)
        results[track] = {gamma: train_car2(track, car1_q, gamma) for gamma in GAMMAS}
    plot_learning_curves(results)
    plot_episodes(results)
    plt.show()


if __name__ == "__main__":
    main()