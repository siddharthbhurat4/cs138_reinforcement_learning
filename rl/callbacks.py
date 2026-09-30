"""Hooks into the training loop: logging, recording data for plots, early stopping...

Write your own by subclassing Callback and overriding only what you need.
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from rl.core import Transition

if TYPE_CHECKING:
    from rl.runner import EpisodeStats, Runner


class Callback:
    def on_train_start(self, runner: "Runner") -> None: ...
    def on_episode_start(self, runner: "Runner", observation: Any) -> None: ...
    def on_step(self, runner: "Runner", transition: Transition, metrics: dict) -> None: ...
    def on_episode_end(self, runner: "Runner", stats: "EpisodeStats") -> None: ...
    def on_train_end(self, runner: "Runner", history: list) -> None: ...


class CallbackList(Callback):
    """Fans every hook out to a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = list(callbacks or [])

    def _call(self, hook: str, *args) -> None:
        for cb in self.callbacks:
            getattr(cb, hook)(*args)

    def on_train_start(self, runner):
        self._call("on_train_start", runner)

    def on_episode_start(self, runner, observation):
        self._call("on_episode_start", runner, observation)

    def on_step(self, runner, transition, metrics):
        self._call("on_step", runner, transition, metrics)

    def on_episode_end(self, runner, stats):
        self._call("on_episode_end", runner, stats)

    def on_train_end(self, runner, history):
        self._call("on_train_end", runner, history)


class ProgressPrinter(Callback):
    def __init__(self, every: int = 100, window: int = 100):
        self.every = every
        self.returns: deque = deque(maxlen=window)
        self.lengths: deque = deque(maxlen=window)

    def on_episode_end(self, runner, stats):
        self.returns.append(stats.total_reward)
        self.lengths.append(stats.length)
        if (stats.episode + 1) % self.every == 0:
            print(f"episode {stats.episode + 1:>6} | avg return {np.mean(self.returns):8.3f} "
                  f"| avg length {np.mean(self.lengths):7.1f}")

# class StepRewardRecorder(Callback):
#     """Keeps every per-step reward, e.g. for 'average reward per step' plots."""

#     def __init__(self):
#         self.episodes = []
#         self.optimal_action = []

#     def on_episode_start(self, runner, observation):
#         self.episodes.append([])
#         self.optimal_action.append([])

#     def on_step(self, runner, transition, metrics):
#         self.episodes[-1].append(transition.reward)
#         self.optimal_action[-1].append(transition.info["is_optimal"])

