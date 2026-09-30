"""Plain data containers passed between environment, agent and runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """What an Environment returns from step()."""
    observation: Any
    reward: float
    terminated: bool          # reached a terminal state (no bootstrapping)
    truncated: bool = False   # cut off by a time limit (do bootstrap)
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass
class Transition:
    """(s, a, r, s', done) tuple handed to Agent.update()."""
    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool = False
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
