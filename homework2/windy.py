"""Wind for the racetrack, as a subclass of RaceTrack, so the existing environment code is unchanged.

In windy cells, before the car makes its own move, the wind may push it one cell to the right
(with probability wind_prob). If the push takes the car into a wall or off the grid, it is a crash:
the car restarts on the start line with zero velocity and the agent's action is not applied that step.
If the push takes the car onto the finish line, the episode ends. Otherwise the car then moves as usual.

    from homework2.windy import WindyRaceTrack, staircase_wind
    grid = make_grid(STAIRCASE)
    env = WindyRaceTrack(grid=grid, grid_size=grid.shape[1], observation_space=(grid.shape[1], grid.shape[0], 5, 5),
                         max_episodes=1000, epsilon_vel_noise=0.1, seed=0,
                         wind_x=staircase_wind(grid), wind_prob=0.5)
"""
import numpy as np

from rl.core import StepResult
# EDIT if your Part 1 code lives in a different module
from homework2.main import FINISH, OFF_TRACK, START, TRACK, RaceTrack


def staircase_wind(grid, strength=1):
    """Wind blowing to the right in the three vertical sections of STAIRCASE (start line excluded)."""
    wind = np.zeros_like(grid)
    for x0, x1, y0, y1 in ((0, 2, 1, 6), (5, 7, 4, 11), (10, 12, 9, 15)):
        wind[y0:y1 + 1, x0:x1 + 1] = strength
    wind[grid != TRACK] = 0
    return wind


class WindyRaceTrack(RaceTrack):
    def __init__(self, *args, wind_x=None, wind_prob=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.wind_x = np.zeros_like(self.grid) if wind_x is None else np.asarray(wind_x)   # cells to push right, per cell
        self.wind_prob = wind_prob

    def _step(self, action):
        push = int(self.wind_x[self.y, self.x])
        if push > 0 and self.np_random.random() < self.wind_prob:
            for x in range(self.x + 1, self.x + push + 1):
                if x >= self.grid.shape[1] or self.grid[self.y, x] == OFF_TRACK:   # blown into a wall
                    self.y, self.x = self.np_random.choice(np.argwhere(self.grid == START))
                    self.vx, self.vy = 0, 0
                    return StepResult(observation=(self.x, self.y, self.vx, self.vy), reward=-1,
                                      terminated=False, info={"wind": True})
                self.x = x
                if self.grid[self.y, x] == FINISH:                                 # blown onto the finish
                    return StepResult(observation=(self.x, self.y, self.vx, self.vy), reward=0,
                                      terminated=True, info={"wind": True})
        return super()._step(action)