"""Base Environment class.

Subclasses implement `_reset()` and `_step()`. The public `reset()` / `step()`
wrap them to handle seeding and time limits in one place (template method).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rl.core.spaces import Space
from rl.core.types import StepResult


class Environment(ABC):
    observation_space: Space
    action_space: Space

    def __init__(self, seed: Any = None, max_episode_steps: int | None = None):
        # NOTE: subclasses should define observation_space / action_space
        # BEFORE calling super().__init__() so the spaces get seeded too.
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self.seed(seed)

    # ---- public API (don't override) ----------------------------------
    def seed(self, seed: Any = None) -> None:
        env_ss, obs_ss, act_ss = np.random.SeedSequence(seed).spawn(3)
        self.np_random = np.random.default_rng(env_ss)
        for space, ss in ((getattr(self, "observation_space", None), obs_ss),
                          (getattr(self, "action_space", None), act_ss)):
            if space is not None:
                space.seed(ss)

    def reset(self, seed: Any = None) -> tuple[Any, dict]:
        if seed is not None:
            self.seed(seed)
        self._elapsed_steps = 0
        return self._reset()

    def step(self, action: Any) -> StepResult:
        result = self._step(action)
        self._elapsed_steps += 1
        if (self.max_episode_steps is not None
                and self._elapsed_steps >= self.max_episode_steps
                and not result.terminated):
            result.truncated = True
        return result

    # ---- to implement ---------------------------------------------------
    @abstractmethod
    def _reset(self) -> tuple[Any, dict]:
        """Return (initial_observation, info)."""

    @abstractmethod
    def _step(self, action: Any) -> StepResult:
        """Apply action, return StepResult (leave `truncated` to the base class)."""

    # ---- optional -------------------------------------------------------
    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
