"""Example 5 - Continuous observations (Box space) and extending an agent by inheritance.

    python -m examples.ex5_continuous_observations
"""
import dataclasses

import numpy as np

from examples.ex3_learning_agent import QLearningAgent
from rl.core import Box, Discrete, Environment, StepResult
from rl.runner import Runner


class NoisyLine(Environment):
    """Position x in [0, 1]. Actions: 0 = left, 1 = right, each moves ~0.1 with noise.
    Reaching x >= 0.95 ends the episode with +1; every other step costs -0.05.
    """

    def __init__(self, noise=0.03, seed=None):
        self.noise = noise
        self.observation_space = Box(low=0.0, high=1.0, shape=(1,))   # a 1-D continuous vector
        self.action_space = Discrete(2)
        super().__init__(seed=seed, max_episode_steps=100)

    def _reset(self):
        self.x = self.np_random.uniform(0.0, 0.3)
        return np.array([self.x], dtype=np.float32), {}

    def _step(self, action):
        move = 0.1 if action == 1 else -0.1
        self.x = float(np.clip(self.x + move + self.np_random.normal(0, self.noise), 0.0, 1.0))
        obs = np.array([self.x], dtype=np.float32)
        if self.x >= 0.95:
            return StepResult(obs, 1.0, terminated=True)
        return StepResult(obs, -0.05, terminated=False)


class BinnedQLearning(QLearningAgent):
    """Reuses QLearningAgent unchanged by converting continuous observations into bins."""

    def __init__(self, observation_space: Box, action_space, n_bins=10, **kwargs):
        low, high = float(observation_space.low[0]), float(observation_space.high[0])
        self.edges = np.linspace(low, high, n_bins + 1)[1:-1]     # inner bin boundaries
        super().__init__(Discrete(n_bins), action_space, **kwargs)  # the parent sees a Discrete space

    def to_bin(self, obs):
        return int(np.digitize(obs[0], self.edges))

    def act(self, obs):
        return super().act(self.to_bin(obs))

    def update(self, t):
        binned = dataclasses.replace(t, obs=self.to_bin(t.obs), next_obs=self.to_bin(t.next_obs))
        return super().update(binned)


def main():
    env = NoisyLine(seed=0)
    print("Observation space:", env.observation_space, "| sample:", env.observation_space.sample())

    agent = BinnedQLearning(env.observation_space, env.action_space, n_bins=10,
                            alpha=0.2, eps_decay_steps=2000, seed=0)
    runner = Runner(env, agent)

    print("before training:", runner.evaluate(num_episodes=50))
    runner.train(num_episodes=500)
    print("after training: ", runner.evaluate(num_episodes=50))
    print("greedy action per bin (1 = right):", agent.greedy_policy())


if __name__ == "__main__":
    main()
