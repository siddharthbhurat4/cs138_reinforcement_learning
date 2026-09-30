"""Example 1 - Writing an Environment and using it by hand (no agent, no runner).

Run from the reinforcement_learning/ folder:
    python -m examples.ex1_environment
"""
from rl.core import Discrete, Environment, StepResult


class GridWalk(Environment):
    """A size x size grid. Start top-left, goal bottom-right, one pit.

    Actions: 0 = up, 1 = right, 2 = down, 3 = left.
    Observation: the cell index, row * size + col.
    Rewards: +1 at the goal, -1 in the pit (both end the episode), -0.01 per other step.
    With probability `slip` a random action is taken instead of the chosen one.
    """

    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, size=4, slip=0.0, pit=(1, 2), max_episode_steps=50, seed=None):
        self.size, self.slip, self.pit = size, slip, pit
        self.goal = (size - 1, size - 1)
        # 1) define the spaces ...
        self.observation_space = Discrete(size * size)
        self.action_space = Discrete(4)
        # 2) ... then let the base class set up seeding and the time limit
        super().__init__(seed=seed, max_episode_steps=max_episode_steps)

    def _reset(self):
        self.pos = (0, 0)
        return self._obs(), {}

    def _step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.np_random.random() < self.slip:          # use self.np_random for all env randomness
            action = int(self.np_random.integers(4))
        dr, dc = self.MOVES[action]
        row = min(max(self.pos[0] + dr, 0), self.size - 1)   # walls: stay inside the grid
        col = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (row, col)

        if self.pos == self.goal:
            return StepResult(self._obs(), 1.0, terminated=True, info={"outcome": "goal"})
        if self.pos == self.pit:
            return StepResult(self._obs(), -1.0, terminated=True, info={"outcome": "pit"})
        return StepResult(self._obs(), -0.01, terminated=False)

    def _obs(self):
        return self.pos[0] * self.size + self.pos[1]

    def render(self):
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = (r, c)
                row.append("A" if cell == self.pos else "G" if cell == self.goal
                           else "X" if cell == self.pit else ".")
            print(" ".join(row))
        print()


def main():
    env = GridWalk(seed=0)
    print("Observation space:", env.observation_space, "| Action space:", env.action_space)

    # --- 1. Step through an episode by hand -------------------------------------
    print("\n1) Manual steps: right, right, down (straight into the pit)")
    obs, info = env.reset()
    env.render()
    for action in [1, 1, 2]:
        result = env.step(action)
        print(f"action={action} -> obs={result.observation}, reward={result.reward}, "
              f"terminated={result.terminated}, info={result.info}")
        env.render()
        if result.done:
            break

    # --- 2. A random rollout until the episode ends ------------------------------
    print("2) Random rollout")
    env.reset()
    total, steps = 0.0, 0
    while True:
        result = env.step(env.action_space.sample())
        total += result.reward
        steps += 1
        if result.done:
            break
    print(f"{steps} steps, return {total:.2f}, terminated={result.terminated}, "
          f"truncated={result.truncated}, info={result.info}")

    # --- 3. Time limit: terminated vs truncated ----------------------------------
    print("\n3) Time limit of 3 steps, walking into the left wall")
    short_env = GridWalk(max_episode_steps=3, seed=0)
    short_env.reset()
    for i in range(3):
        result = short_env.step(3)
        print(f"step {i + 1}: terminated={result.terminated}, truncated={result.truncated}")

    # --- 4. Seeding makes randomness reproducible --------------------------------
    print("\n4) Seeding (slippery grid, always pressing 'right')")

    def rollout(seed):
        env = GridWalk(slip=0.5, seed=seed)
        env.reset()
        cells = []
        for _ in range(6):
            result = env.step(1)
            cells.append(result.observation)
            if result.done:
                break
        return cells

    print("seed 42:", rollout(42), "| seed 42 again:", rollout(42), "| seed 7:", rollout(7))


if __name__ == "__main__":
    main()
