"""The interface between Environment and Agent.

The Runner owns the interaction loop so that neither side knows about the other:
    obs -> agent.act -> env.step -> Transition -> agent.update -> callbacks
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from rl.callbacks import Callback, CallbackList
from rl.core import Agent, Environment, Transition


@dataclass
class EpisodeStats:
    episode: int
    total_reward: float
    length: int
    terminated: bool
    metrics: dict = field(default_factory=dict)  # per-step metrics averaged over the episode


class Runner:
    def __init__(self, env: Environment, agent: Agent, callbacks: list[Callback] | None = None):
        self.env, self.agent = env, agent
        self.callbacks = CallbackList(callbacks)
        self.episode = 0       # training episodes completed
        self.global_step = 0   # training steps completed
        self.stop_training = False

    def run_episode(self, train: bool = True, max_steps: int | None = None,
                    render: bool = False) -> EpisodeStats:
        agent, env = self.agent.train() if train else self.agent.eval(), self.env
        cbs = self.callbacks if train else CallbackList()

        obs, _ = env.reset()
        agent.on_episode_start(obs)
        cbs.on_episode_start(self, obs)
        total, length, terminated = 0.0, 0, False
        metric_sums: dict[str, float] = defaultdict(float)

        while max_steps is None or length < max_steps:
            if render:
                env.render()
            action = agent.act(obs)
            result = env.step(action)
            t = Transition(obs, action, result.reward, result.observation,
                           result.terminated, result.truncated, result.info)
            metrics = agent.update(t) if train else {}
            for k, v in metrics.items():
                metric_sums[k] += v
            cbs.on_step(self, t, metrics)

            total += result.reward
            length += 1
            if train:
                self.global_step += 1
            obs = result.observation
            if result.done:
                terminated = result.terminated
                break

        agent.on_episode_end()
        stats = EpisodeStats(self.episode, total, length, terminated,
                             {k: v / max(length, 1) for k, v in metric_sums.items()})
        if train:
            cbs.on_episode_end(self, stats)
            self.episode += 1
        return stats

    def train(self, num_episodes: int, max_steps: int | None = None) -> list[EpisodeStats]:
        self.stop_training = False
        self.callbacks.on_train_start(self)
        history = []
        for _ in range(num_episodes):
            history.append(self.run_episode(train=True, max_steps=max_steps))
            if self.stop_training:
                break
        self.callbacks.on_train_end(self, history)
        return history

    def evaluate(self, num_episodes: int = 10, max_steps: int | None = None,
                 render: bool = False) -> dict[str, float]:
        stats = [self.run_episode(train=False, max_steps=max_steps, render=render)
                 for _ in range(num_episodes)]
        returns = np.array([s.total_reward for s in stats])
        return {"mean_return": float(returns.mean()), "std_return": float(returns.std()),
                "mean_length": float(np.mean([s.length for s in stats])),
                "success_rate": float(np.mean([s.terminated for s in stats]))}
