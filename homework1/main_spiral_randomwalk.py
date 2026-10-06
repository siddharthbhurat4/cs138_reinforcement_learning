import os

import numpy as np

from rl.callbacks import Callback
from rl.core import Agent, Discrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed
from homework1.plotting_spiral_randomwalk import plot_action_means, plot_bandit_results
from homework1.results_io import save_results


# environment
class TenArmedBandit(Environment):
    def __init__(self, n_action: int = 10, max_steps=10000, rewards_standarddeviation=0.05,
                 drift="spiral",
                 randomwalk_mean=0.0, randomwalk_standarddeviation=0.01,
                 spiral_start_radius=1.0, spiral_growth=0.0, spiral_steps_per_turn=2000,
                 seed=0):
        if drift not in ("random_walk", "spiral"):
            raise ValueError(f"drift must be 'random_walk' or 'spiral', got {drift!r}")
        self.n_observation = 1
        self.n_action = n_action
        self.observation_space = Discrete(self.n_observation)   # define spaces first...
        self.action_space = Discrete(n_action)
        self.rewards_standarddeviation = rewards_standarddeviation
        self.drift = drift

        # random-walk settings
        self.randomwalk_mean = randomwalk_mean
        self.randomwalk_standarddeviation = randomwalk_standarddeviation

        # spiral settings
        self.spiral_start_radius = spiral_start_radius          # a
        self.spiral_growth = spiral_growth                      # b: radius gained per radian turned
        self.spiral_omega = 2 * np.pi / spiral_steps_per_turn   # angle moved per step
        self.spiral_start_angles = 2 * np.pi * np.arange(n_action) / n_action

        self._reset_state()
        # we get the random number generator from our environment base class: self.np_random
        super().__init__(seed=seed, max_episode_steps=max_steps)  # ...then call super()

    # state
    def _reset_state(self):
        self.t = 0
        if self.drift == "spiral":
            self._update_spiral_means()
        else:
            self.reward_means = np.zeros(self.n_action)
            self.spiral_positions = None

    def _reset(self):
        self._reset_state()
        return 0, {}

    # drift: random walk
    def _rewardrandomwalk(self):
        random_movement = self.np_random.normal(loc=self.randomwalk_mean,
                                                scale=self.randomwalk_standarddeviation,
                                                size=self.n_action)
        self.reward_means += random_movement

    # drift: spiral
    def _update_spiral_means(self):
        """Recompute every action's point on the spiral and its mean from the current step t."""
        turned = self.spiral_omega * self.t                      # total angle moved so far
        radius = self.spiral_start_radius + self.spiral_growth * turned
        angles = self.spiral_start_angles + turned
        x, y = radius * np.cos(angles), radius * np.sin(angles)
        self.spiral_positions = np.stack([x, y], axis=1)         # shape (n_action, 2), for plotting
        self.reward_means = y                                    # mean = height on the spiral

    def _move_along_spiral(self):
        self.t += 1
        self._update_spiral_means()

    # step
    def _step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        # terminated is always False: a bandit never ends; the base class truncates at max_steps
        thisStepReward = float(self.np_random.normal(loc=self.reward_means[action],
                                                     scale=self.rewards_standarddeviation))
        thisStepBestAction = int(np.argmax(self.reward_means))
        # tie-aware: any action whose mean equals the best counts as optimal
        isOptimal = bool(np.isclose(self.reward_means[action], self.reward_means.max()))

        if self.drift == "spiral":
            self._move_along_spiral()
        else:
            self._rewardrandomwalk()

        return StepResult(observation=0, reward=thisStepReward, terminated=False,
                          info={"optimal_action": thisStepBestAction, "is_optimal": isOptimal})


# agent
class Jumbo(Agent):
    def __init__(self, observation_space, action_space, alpha=0.1, epsilon=0.1, seed=None,
                 action_value_update_method="sample_averaging"):
        super().__init__(observation_space, action_space, seed)
        self.alpha, self.epsilon = alpha, epsilon
        self.action_value = np.zeros(self.action_space.n)
        self.reward_store = [[] for _ in range(self.action_space.n)]
        self.action_taken = np.zeros(self.action_space.n)
        self.action_value_updated_method = action_value_update_method
        self.temperature = 0.1

    def act(self, obs):
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_space.n))
        # best = np.flatnonzero(self.action_value == self.action_value.max())
        # return int(self.rng.choice(best))  # random tie-breaking

        #softmax action selection method
        #divide by temperature
        self.action_value_exp_over_temp = np.exp(self.action_value/self.temperature)
        self.temperature -= 0
        self.action_value_exp_over_temp_sum = sum(self.action_value_exp_over_temp)
        self.action_value_softmax = self.action_value_exp_over_temp / self.action_value_exp_over_temp_sum
        best = np.flatnonzero(self.action_value_softmax == self.action_value_softmax.max())
        return int(self.rng.choice(self.action_space.n, p=self.action_value_softmax))  # random tie-breaking


    def update(self, t):
        # sample averaging: keeps every reward
        if self.action_value_updated_method == "sample_averaging":
            self.reward_store[t.action].append(t.reward)
            self.action_value[t.action] = sum(self.reward_store[t.action]) / len(self.reward_store[t.action])
        # incremental averaging: Q_n+1 = Q_n + (1/n) * (R_n - Q_n)
        elif self.action_value_updated_method == "incremental_average":
            self.action_taken[t.action] += 1
            self.action_value[t.action] += (1 / self.action_taken[t.action]) * (t.reward - self.action_value[t.action])
        # constant step size: Q_n+1 = Q_n + alpha * (R_n - Q_n)
        elif self.action_value_updated_method == "step_size_method":
            self.action_value[t.action] += self.alpha * (t.reward - self.action_value[t.action])
        else:
            raise ValueError(f"Unknown update method: {self.action_value_updated_method!r}")
        return {}

    def on_episode_start(self, observation) -> None:
        self.action_taken = np.zeros(self.action_space.n)
        self.action_value = np.zeros(self.action_space.n)
        self.reward_store = [[] for _ in range(self.action_space.n)]


class StepRewardRecorder(Callback):
    """Keeps every per-step reward and whether the chosen action was optimal."""

    def __init__(self):
        self.episodes = []
        self.optimal_action = []

    def on_episode_start(self, runner, observation):
        self.episodes.append([])
        self.optimal_action.append([])

    def on_step(self, runner, transition, metrics):
        self.episodes[-1].append(transition.reward)
        self.optimal_action[-1].append(transition.info["is_optimal"])


#settings
N_STEPS = 10000
N_EPISODES = 1000
EPSILON = 0.1
ALPHA = 0.1
WINDOW = 1
SEED = 0

DRIFT = "spiral"            # "spiral" or "random_walk"
SPIRAL_START_RADIUS = 1.0   # a
SPIRAL_GROWTH = 0.03         # b: 0 = circle; ~0.03 grows the radius from 1 to ~2 over 10,000 steps
SPIRAL_STEPS_PER_TURN = 2000

RESULTS_FILE = f"results/hw1_{DRIFT}.npz"

METHODS = {                 # plot label -> action_value_update_method passed to the agent
    "sample average": "sample_averaging",
    "incremental average": "incremental_average",
    f"constant step size (α={ALPHA})": "step_size_method",
}


#experiment
def make_env():
    return TenArmedBandit(n_action=10, max_steps=N_STEPS, rewards_standarddeviation=0.05,
                          drift=DRIFT,
                          randomwalk_mean=0.0, randomwalk_standarddeviation=0.01,
                          spiral_start_radius=SPIRAL_START_RADIUS, spiral_growth=SPIRAL_GROWTH,
                          spiral_steps_per_turn=SPIRAL_STEPS_PER_TURN, seed=SEED)


def trace_action_means(env, n_steps):
    env.reset()
    means, positions = [], []
    for _ in range(n_steps):
        means.append(env.reward_means.copy())          # the means used for this step's reward
        if env.spiral_positions is not None:
            positions.append(env.spiral_positions.copy())
        env.step(0)
    return np.array(means), (np.array(positions) if positions else None)


def run_method(method):
    env = make_env()   # fresh env, same seed -> every method faces the same drift
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, alpha=ALPHA,
                  seed=SEED, action_value_update_method=method)
    recorder = StepRewardRecorder()
    Runner(env, agent, callbacks=[recorder]).train(num_episodes=N_EPISODES)
    return recorder.episodes, recorder.optimal_action


def main():
    set_global_seed(SEED)

    # 1) Look at the environment before training
    means, positions = trace_action_means(make_env(), N_STEPS)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    np.save(RESULTS_FILE.replace(".npz", "_action_means.npy"), means)
    plot_action_means(means, positions,
                      title=f"True action means ({DRIFT}, {SPIRAL_STEPS_PER_TURN} steps per turn, "
                            f"growth b={SPIRAL_GROWTH})" if DRIFT == "spiral" else "True action means (random walk)")

    # 2) Train all three methods and save the results
    results = {label: run_method(method) for label, method in METHODS.items()}
    save_results(results, RESULTS_FILE, settings={
        "n_steps": N_STEPS, "n_runs": N_EPISODES, "epsilon": EPSILON, "alpha": ALPHA, "seed": SEED,
        "drift": DRIFT, "spiral_start_radius": SPIRAL_START_RADIUS, "spiral_growth": SPIRAL_GROWTH,
        "spiral_steps_per_turn": SPIRAL_STEPS_PER_TURN,
    })

    # 3) Summary table: performance over the last 1,000 steps
    print(f"{'method':32s} {'avg reward':>11s} {'% optimal':>10s}   (last 1000 steps)")
    for label, (rewards, is_optimal) in results.items():
        rewards, is_optimal = np.array(rewards), np.array(is_optimal)
        print(f"{label:32s} {rewards[:, -1000:].mean():11.3f} {100 * is_optimal[:, -1000:].mean():10.1f}")

    # Sanity check: sample and incremental averaging must agree
    sample = np.array(results["sample average"][0])
    incremental = np.array(results["incremental average"][0])
    print(f"\nmax |reward difference| sample vs incremental: {np.max(np.abs(sample - incremental)):.2e}")

    # 4) Plot
    title = ("Spiral bandit" if DRIFT == "spiral" else "Random-walk bandit")
    plot_bandit_results(results, window=WINDOW, layout="overlay", title=title)


if __name__ == "__main__":
    main()