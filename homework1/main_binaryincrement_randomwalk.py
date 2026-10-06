import os

import numpy as np

from rl.callbacks import Callback
from rl.core import Agent, Discrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed
from homework1.plotting_binaryincrement_randomwalk import plot_action_means, plot_bandit_results
from homework1.results_io import save_results


# environment
class BinaryIncrementBandit(Environment):
    def __init__(self, n_action: int = 10, max_steps=10000, rewards_standarddeviation=0.05,
                 mean_scale=1.0, steps_per_increment=1, seed=0):
        self.n_observation = 1
        self.n_action = n_action
        self.observation_space = Discrete(self.n_observation)
        self.action_space = Discrete(n_action)
        self.rewards_standarddeviation = rewards_standarddeviation
        self.mean_scale = mean_scale
        self.steps_per_increment = steps_per_increment
        self.place_values = 2 ** np.arange(n_action)
        self.t = 0
        self._update_means()
        super().__init__(seed=seed, max_episode_steps=max_steps)

    def _update_means(self):
        """Action k's mean = bit k of the counter"""
        counter = self.t // self.steps_per_increment
        self.reward_means = self.mean_scale * ((counter // self.place_values) % 2)

    def _reset(self):
        self.t = 0
        self._update_means()
        return 0, {}

    def _step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        thisStepReward = float(self.np_random.normal(loc=self.reward_means[action],
                                                     scale=self.rewards_standarddeviation))
        thisStepBestAction = int(np.argmax(self.reward_means))
        isOptimal = bool(np.isclose(self.reward_means[action], self.reward_means.max()))
        self.t += 1
        self._update_means()
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
        self.T = 2000
        self.t = 0
        self.epsilon_start = 0.1
        self.epsilon_min = 0.05
        self.temp_start = 0.5
        self.temp_min = 0.15

    def act(self, obs):
        #Epsilon greedy action selection
        # if self.training and self.rng.random() < self.epsilon:
        #     return int(self.rng.integers(self.action_space.n))
        # best = np.flatnonzero(self.action_value == self.action_value.max())
        # return int(self.rng.choice(best))  # random tie-breaking

        #softmax action selection method
        #divide by temperature
        self.temperature = max(self.temp_min , self.temp_start - (self.temp_start - self.temp_min) * self.t / self.T)
        self.action_value_exp_over_temp = np.exp(self.action_value/self.temperature)
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

MEAN_SCALE = 1.0
STEPS_PER_INCREMENT = 1

RESULTS_FILE = "results/hw1_binary.npz"

METHODS = {
    "sample average": "sample_averaging",
    "incremental average": "incremental_average",
    f"constant step size (α={ALPHA})": "step_size_method",
}


#experiment
def make_env():
    return BinaryIncrementBandit(n_action=10, max_steps=N_STEPS, rewards_standarddeviation=0.05,
                                 mean_scale=MEAN_SCALE, steps_per_increment=STEPS_PER_INCREMENT, seed=SEED)


def trace_action_means(env, n_steps):
    """True means at every step of one run"""
    env.reset()
    means = []
    for _ in range(n_steps):
        means.append(env.reward_means.copy())
        env.step(0)
    return np.array(means)


def run_method(method):
    env = make_env()   # fresh env, same seed -> every method faces the same sequence
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, alpha=ALPHA,
                  seed=SEED, action_value_update_method=method)
    recorder = StepRewardRecorder()
    Runner(env, agent, callbacks=[recorder]).train(num_episodes=N_EPISODES)
    return recorder.episodes, recorder.optimal_action


def main():
    set_global_seed(SEED)

    # 1) Look at the environment before training
    means = trace_action_means(make_env(), N_STEPS)
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    np.save(RESULTS_FILE.replace(".npz", "_action_means.npy"), means)
    plot_action_means(means)

    # 2) Train all three methods and save the results
    results = {label: run_method(method) for label, method in METHODS.items()}
    save_results(results, RESULTS_FILE, settings={
        "n_steps": N_STEPS, "n_runs": N_EPISODES, "epsilon": EPSILON, "alpha": ALPHA, "seed": SEED,
        "drift": "binary_increment", "mean_scale": MEAN_SCALE, "steps_per_increment": STEPS_PER_INCREMENT,
    })

    # 3) Summary table: performance over the last 1,000 steps
    print(f"{'method':32s} {'avg reward':>11s} {'% optimal':>10s}   (last 1000 steps)")
    for label, (rewards, is_optimal) in results.items():
        rewards, is_optimal = np.array(rewards), np.array(is_optimal)
        print(f"{label:32s} {rewards[:, -1000:].mean():11.3f} {100 * is_optimal[:, -1000:].mean():10.1f}")

    # 4) Plot
    plot_bandit_results(results, window=WINDOW, layout="overlay", title="Binary-increment bandit")


if __name__ == "__main__":
    main()