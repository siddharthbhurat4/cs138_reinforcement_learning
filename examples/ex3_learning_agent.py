"""Example 3 - A learning agent: training, metrics, comparing settings, plotting, save/load.

    python -m examples.ex3_learning_agent            # add --plot to see learning curves
"""
import argparse
import os

import numpy as np

from examples.ex1_environment import GridWalk
from rl.callbacks import ProgressPrinter
from rl.core import Agent
from rl.runner import Runner
from rl.utils import moving_average, plot_curves, set_global_seed


class QLearningAgent(Agent):
    """Tabular Q-learning with epsilon decaying linearly over training steps."""

    def __init__(self, observation_space, action_space, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=3000, seed=None):
        super().__init__(observation_space, action_space, seed)
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.q = np.zeros((observation_space.n, action_space.n))
        self.steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, obs):
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_space.n))
        best = np.flatnonzero(self.q[obs] == self.q[obs].max())
        return int(self.rng.choice(best))

    def update(self, t):
        # no bootstrapping from real terminal states; truncated states still bootstrap
        target = t.reward + self.gamma * (not t.terminated) * self.q[t.next_obs].max()
        td_error = target - self.q[t.obs, t.action]
        self.q[t.obs, t.action] += self.alpha * td_error
        self.steps += 1
        return {"td_error": abs(td_error), "epsilon": self.epsilon}   # averaged per episode by the runner

    def greedy_policy(self):
        return np.argmax(self.q, axis=1)

    # save/load only needs to know what state matters
    def state_dict(self):
        return {"q": self.q.copy(), "steps": self.steps}

    def load_state_dict(self, state):
        self.q, self.steps = state["q"].copy(), state["steps"]


def print_policy(env, policy):
    arrows = "↑→↓←"
    for r in range(env.size):
        row = []
        for c in range(env.size):
            cell = (r, c)
            row.append("G" if cell == env.goal else "X" if cell == env.pit
                       else arrows[policy[r * env.size + c]])
        print(" ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    set_global_seed(0)

    # --- 1. Train and compare two exploration settings --------------------------
    settings = {
        "epsilon decays 1.0 -> 0.05": dict(eps_start=1.0, eps_end=0.05),
        "constant epsilon 0.1": dict(eps_start=0.1, eps_end=0.1),
    }
    curves = {}
    for label, kwargs in settings.items():
        print(f"\n=== {label} ===")
        env = GridWalk(slip=0.1, seed=0)
        agent = QLearningAgent(env.observation_space, env.action_space, seed=0, **kwargs)
        runner = Runner(env, agent, callbacks=[ProgressPrinter(every=250)])
        history = runner.train(num_episodes=args.episodes)      # list of EpisodeStats

        curves[label] = [h.total_reward for h in history]
        td = [h.metrics["td_error"] for h in history]
        print(f"mean |TD error|: first 50 episodes {np.mean(td[:50]):.3f}, last 50 {np.mean(td[-50:]):.3f}")
        print(f"final epsilon {history[-1].metrics['epsilon']:.3f}, total steps {runner.global_step}")
        print("greedy evaluation:", runner.evaluate(num_episodes=100))

    # --- 2. Inspect what was learned (last agent) --------------------------------
    print("\nGreedy policy of the last agent:")
    print_policy(env, agent.greedy_policy())
    smooth = moving_average(curves["constant epsilon 0.1"], window=50)
    print(f"smoothed training return: start {smooth[0]:.3f} -> end {smooth[-1]:.3f}")

    # --- 3. Save and load ------------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)          # checkpoints/ is in .gitignore
    agent.save("checkpoints/q_agent.pkl")
    restored = QLearningAgent(env.observation_space, env.action_space)
    restored.load("checkpoints/q_agent.pkl")
    print("\nrestored agent identical:", np.array_equal(agent.q, restored.q))
    print("restored agent evaluation:", Runner(env, restored).evaluate(num_episodes=100))

    # --- 4. Plot ---------------------------------------------------------------
    if args.plot:
        plot_curves(curves, window=50, title="Q-learning on slippery GridWalk")
        # plot_curves(curves, window=50, save_path="results/q_learning.png")  # for a report


if __name__ == "__main__":
    main()
