"""Homework template - copy this folder for each assignment:

    cp -r homework_template homework1
    python -m homework1.main          # run from the reinforcement_learning/ folder

Everything below is an example to replace with your own env and agent.
As the homework grows, split them into env.py / agent.py in this folder.
"""
import numpy as np

from rl.callbacks import ProgressPrinter
from rl.core import Agent, Discrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed


# ---------------------------------------------------------------- environment
class Corridor(Environment):
    """Start at cell 0, reach cell n-1. Actions: 0 = left, 1 = right."""

    def __init__(self, n: int = 8, seed=None):
        self.n = n
        self.observation_space = Discrete(n)   # define spaces first...
        self.action_space = Discrete(2)
        super().__init__(seed=seed, max_episode_steps=5 * n)  # ...then call super()

    def _reset(self):
        self.pos = 0
        return self.pos, {}

    def _step(self, action):
        self.pos = int(np.clip(self.pos + (1 if action == 1 else -1), 0, self.n - 1))
        at_goal = self.pos == self.n - 1
        return StepResult(observation=self.pos, reward=1.0 if at_goal else -0.1, terminated=at_goal)


# ---------------------------------------------------------------------- agent
class QLearning(Agent):
    """Minimal tabular Q-learning, showing where each piece goes."""

    def __init__(self, observation_space, action_space, alpha=0.5, gamma=0.99, epsilon=0.1, seed=None):
        super().__init__(observation_space, action_space, seed)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.q = np.zeros((observation_space.n, action_space.n))

    def act(self, obs):
        if self.training and self.rng.random() < self.epsilon:
            return self.action_space.sample()
        best = np.flatnonzero(self.q[obs] == self.q[obs].max())
        return int(self.rng.choice(best))  # random tie-breaking

    def update(self, t):
        target = t.reward + self.gamma * (not t.terminated) * self.q[t.next_obs].max()
        td_error = target - self.q[t.obs, t.action]
        self.q[t.obs, t.action] += self.alpha * td_error
        return {"td_error": abs(td_error)}


# ----------------------------------------------------------------- experiment
def main():
    set_global_seed(0)
    env = Corridor(n=8, seed=0)
    agent = QLearning(env.observation_space, env.action_space, seed=0)

    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=50)])
    history = runner.train(num_episodes=200)
    print("Evaluation:", runner.evaluate(num_episodes=10))

    # from rl.utils import plot_curves
    # plot_curves({"Q-learning": [h.total_reward for h in history]}, window=10)


if __name__ == "__main__":
    main()
