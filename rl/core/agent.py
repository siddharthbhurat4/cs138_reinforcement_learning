"""Base Agent class. An agent only needs to know the spaces, never the env itself."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rl.core.spaces import Space
from rl.core.types import Transition


class Agent(ABC):
    def __init__(self, observation_space: Space, action_space: Space, seed: Any = None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)
        self.training = True

    # ---- mode switching ---------------------------------------------------
    def train(self) -> "Agent":
        self.training = True
        return self

    def eval(self) -> "Agent":
        self.training = False
        return self

    # ---- core interface ---------------------------------------------------
    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Choose an action. Should explore if self.training, else act greedily."""

    def update(self, transition: Transition) -> dict[str, float]:
        """Learn from one transition. Return any metrics you want logged."""
        return {}

    # ---- lifecycle hooks (optional) ---------------------------------------
    def on_episode_start(self, observation: Any) -> None:
        pass

    def on_episode_end(self) -> None:
        pass

    # ---- persistence ------------------------------------------------------
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self.load_state_dict(pickle.load(f))
