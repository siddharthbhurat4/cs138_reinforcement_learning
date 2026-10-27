"""Part 3: windy staircase. Trains the Part 1 agent on the staircase with wind probability
0.25, 0.5 and 1.0, then shows greedy routes and the learning curves. Nothing is saved; save
figures from the plot window.

    python -m homework2.run_part3          # run from the reinforcement_learning/ folder
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from homework2.main import FINISH, OFF_TRACK, START, TRACK, Jumbo
from homework2.racetracks import TRACK_3, make_grid
from homework2.windy import WindyRaceTrack, staircase_wind
from rl.callbacks import ProgressPrinter
from rl.runner import Runner
from rl.utils import moving_average, set_global_seed

WIND_PROBS = [0.25, 0.5, 1.0]
NUM_EPISODES = 10000
NUM_STEPS = 2000        # per-episode step limit
EPSILON = 0.1
GAMMA = 1
SEED = 0
WINDOW = 200            # moving-average window for the learning curves
N_ROUTES = 3            # greedy routes drawn per wind probability
TRACK_COLORS = {TRACK: "#f2f2f2", OFF_TRACK: "#555555", START: "#9ecae1", FINISH: "#a1d99b"}


def make_env(wind_prob, seed=SEED, max_steps=NUM_STEPS):
    grid = make_grid(TRACK_3)
    return WindyRaceTrack(grid=grid, grid_size=grid.shape[1], observation_space=(grid.shape[1], grid.shape[0], 5, 5),
                          max_episodes=max_steps, epsilon_vel_noise=0.1, seed=seed,
                          wind_x=staircase_wind(grid), wind_prob=wind_prob)


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


def train(wind_prob):
    """Train one agent; return its episode lengths and a few greedy routes (the agent is then discarded)."""
    set_global_seed(SEED)
    env = make_env(wind_prob)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, gamma=GAMMA, seed=SEED)
    print(f"\n=== wind_prob = {wind_prob} ===")
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=max(NUM_EPISODES // 5, 1))])
    history = runner.train(NUM_EPISODES)
    print("Evaluation (greedy, 100 episodes):", runner.evaluate(num_episodes=100))
    route_env = make_env(wind_prob, seed=SEED + 1, max_steps=300)
    routes = [greedy_route(route_env, agent) for _ in range(N_ROUTES)]
    return [h.length for h in history], routes


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


def plot_routes(results):
    grid = make_grid(TRACK_3)
    wind = staircase_wind(grid)
    fig, axes = plt.subplots(1, len(WIND_PROBS), figsize=(5.2 * len(WIND_PROBS), 5.6), squeeze=False)
    for ax, p in zip(axes[0], WIND_PROBS):
        draw_track(ax, grid, wind)
        for i, (xs, ys, steps, finished) in enumerate(results[p][1]):
            ax.plot(xs, ys, "-o", color=f"C{i}", markersize=3, linewidth=1.6, alpha=0.85,
                    label=f"route {i + 1}: {steps} steps" + ("" if finished else " (not finished)"))
        ax.set_title(f"wind_prob = {p}")
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Part 3: greedy routes on the windy staircase (wind → pushes right; velocity noise on)")
    fig.tight_layout()


def plot_learning_curves(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in WIND_PROBS:
        lengths = results[p][0]
        smoothed = moving_average(lengths, WINDOW)
        ax.plot(np.arange(len(lengths) - len(smoothed) + 1, len(lengths) + 1), smoothed, label=f"wind_prob = {p}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode length (steps)")
    ax.set_title(f"Part 3: windy staircase learning curves (moving average over {WINDOW} episodes)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()


def main():
    results = {p: train(p) for p in WIND_PROBS}
    plot_routes(results)
    plot_learning_curves(results)
    plt.show()


if __name__ == "__main__":
    main()