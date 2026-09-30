"""Example 2 - Writing simple Agents and running them with the Runner.

    python -m examples.ex2_agents_and_runner
"""
import math

from examples.ex1_environment import GridWalk
from rl.core import Agent
from rl.runner import Runner


class RandomAgent(Agent):
    """The simplest possible agent: only act() is required."""

    def act(self, observation):
        return int(self.rng.integers(self.action_space.n))


class RightThenDown(Agent):
    """A hand-coded policy: go right along the top row, then down the last column."""

    def __init__(self, observation_space, action_space, seed=None):
        super().__init__(observation_space, action_space, seed)
        self.size = int(math.isqrt(observation_space.n))   # grid width, derived from the space

    def act(self, observation):
        col = observation % self.size
        return 1 if col < self.size - 1 else 2   # 1 = right, 2 = down


def main():
    env = GridWalk(seed=0)

    # --- 1. One episode ---------------------------------------------------------
    agent = RandomAgent(env.observation_space, env.action_space, seed=0)
    runner = Runner(env, agent)
    stats = runner.run_episode(train=False)
    print("1) One random episode:", stats)

    # --- 2. Evaluate over many episodes -----------------------------------------
    print("\n2) Evaluation over 200 episodes")
    for agent_cls in [RandomAgent, RightThenDown]:
        for slip in [0.0, 0.2]:
            env = GridWalk(slip=slip, seed=0)
            agent = agent_cls(env.observation_space, env.action_space, seed=0)
            result = Runner(env, agent).evaluate(num_episodes=200)
            print(f"{agent_cls.__name__:14s} slip={slip}: mean return {result['mean_return']:6.3f}, "
                  f"mean length {result['mean_length']:5.1f}, ended by termination {result['success_rate']:.0%}")

    # --- 3. Watch an episode ----------------------------------------------------
    print("\n3) Rendering one episode of RightThenDown (state before each action)")
    env = GridWalk(seed=0)
    Runner(env, RightThenDown(env.observation_space, env.action_space)).run_episode(train=False, render=True)


if __name__ == "__main__":
    main()
