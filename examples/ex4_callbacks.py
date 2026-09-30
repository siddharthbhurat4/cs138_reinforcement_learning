"""Example 4 - Writing your own callbacks.

    python -m examples.ex4_callbacks
"""
import csv
import os
from collections import Counter, deque

import numpy as np

from examples.ex1_environment import GridWalk
from examples.ex3_learning_agent import QLearningAgent
from rl.callbacks import Callback, ProgressPrinter
from rl.runner import Runner


class OutcomeCounter(Callback):
    """Counts how episodes end, using the env's info dict. Override only the hooks you need."""

    def __init__(self):
        self.counts = Counter()
        self._last_info = {}

    def on_step(self, runner, transition, metrics):
        self._last_info = transition.info

    def on_episode_end(self, runner, stats):
        self.counts[self._last_info.get("outcome", "time limit")] += 1


class StepRewardRecorder(Callback):
    """Keeps every per-step reward, e.g. for 'average reward per step' plots."""

    def __init__(self):
        self.episodes = []

    def on_episode_start(self, runner, observation):
        self.episodes.append([])

    def on_step(self, runner, transition, metrics):
        self.episodes[-1].append(transition.reward)


class EarlyStopping(Callback):
    """Stops training once the average return over `window` episodes reaches `threshold`."""

    def __init__(self, threshold, window=100):
        self.threshold = threshold
        self.returns = deque(maxlen=window)

    def on_episode_end(self, runner, stats):
        self.returns.append(stats.total_reward)
        if len(self.returns) == self.returns.maxlen and np.mean(self.returns) >= self.threshold:
            print(f"Early stopping after episode {stats.episode + 1}")
            runner.stop_training = True


class Checkpoint(Callback):
    """Saves the agent every `every` episodes and once more at the end."""

    def __init__(self, path, every=200):
        self.path, self.every = path, every

    def on_episode_end(self, runner, stats):
        if (stats.episode + 1) % self.every == 0:
            runner.agent.save(self.path)

    def on_train_end(self, runner, history):
        runner.agent.save(self.path)


class CSVLogger(Callback):
    """Writes one row per episode - handy for making report plots/tables later."""

    def __init__(self, path):
        self.path = path

    def on_train_start(self, runner):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "return", "length", "td_error", "epsilon"])

    def on_episode_end(self, runner, stats):
        self.writer.writerow([stats.episode, round(stats.total_reward, 4), stats.length,
                              round(stats.metrics.get("td_error", 0.0), 4),
                              round(stats.metrics.get("epsilon", 0.0), 4)])

    def on_train_end(self, runner, history):
        self.file.close()


def main():
    env = GridWalk(slip=0.1, seed=0)
    agent = QLearningAgent(env.observation_space, env.action_space, seed=0)

    outcomes = OutcomeCounter()
    rewards = StepRewardRecorder()
    callbacks = [
        ProgressPrinter(every=200),
        outcomes,
        rewards,
        EarlyStopping(threshold=0.85, window=100),
        Checkpoint("checkpoints/ex4_agent.pkl", every=200),
        CSVLogger("results/ex4_training_log.csv"),    # results/ is in .gitignore
    ]
    history = Runner(env, agent, callbacks=callbacks).train(num_episodes=3000)

    print(f"\ntrained for {len(history)} episodes")
    print("episode outcomes:", dict(outcomes.counts))
    print("steps recorded in first / last episode:", len(rewards.episodes[0]), "/", len(rewards.episodes[-1]))
    print("log written to results/ex4_training_log.csv, checkpoint to checkpoints/ex4_agent.pkl")


if __name__ == "__main__":
    main()
