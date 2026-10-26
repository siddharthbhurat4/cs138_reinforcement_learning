"""Homework
    python -m homework2.main          # run from the reinforcement_learning/ folder
"""
import os

import numpy as np

from rl.callbacks import ProgressPrinter
from rl.core import Agent, Discrete, MultiDiscrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed

from homework2.racetracks import TRACK_1, TRACK_2, TRACK_3, make_grid


ACTIONS = [(0,0), (0,1), (0,-1),
           (1,0), (1,1), (1,-1),
           (-1,0), (-1,1), (-1,-1)]

TRACK, OFF_TRACK, START, FINISH = 0, 1, 2, 3
# ---------------------------------------------------------------- environment
class RaceTrack(Environment):

    def __init__(self, grid, grid_size, observation_space, max_episodes, epsilon_vel_noise=0.1, seed=None):
        self.observation_space = MultiDiscrete(observation_space)   # define spaces first...
        self.action_space = Discrete(len(ACTIONS))

        self.epsilon_vel_noise = epsilon_vel_noise
        #Grid:
        self.grid_size = grid_size
        # grid[y, x]: x = column (increasing right), y = row (increasing UP, y = 0 is the bottom row)
        #bottom left is 0,0 that is x,y
        self.grid = grid
        super().__init__(seed=seed, max_episode_steps=max_episodes)  # ...then call super()

    def _reset(self):
        self.y, self.x = self.np_random.choice(np.argwhere(self.grid == START))
        self.vx = 0
        self.vy = 0
        return (self.x, self.y, self.vx, self.vy), {}

    def _step(self, action):
        dx, dy = ACTIONS[action]
        #velocity noise:
        if self.np_random.random() < self.epsilon_vel_noise:
            dx = 0
            dy = 0

        # velocity after this action
        new_vx = self.vx + dx if 0 <= self.vx + dx < self.observation_space.sizes[2] else self.vx
        new_vy = self.vy + dy if 0 <= self.vy + dy < self.observation_space.sizes[3] else self.vy

        # velocity (0, 0) is only allowed on the start line; ignore elsewhere
        on_start_line = self.grid[self.y, self.x] == START
        if new_vx == 0 and new_vy == 0 and not on_start_line:
            new_vx, new_vy = self.vx, self.vy

        self.vx, self.vy = new_vx, new_vy

        new_x = self.x + self.vx
        new_y = self.y + self.vy

        if self.vx == 0 and self.vy == 0:
            #nothing happened and only allowed at start line
            reward = -1
            at_goal = False

        points = np.round(np.linspace((self.x, self.y), (new_x, new_y), max(self.vx, self.vy) + 1)).astype(int)
        
        for x,y in points[1:]:
            if x<0 or x>=self.grid.shape[1] or y<0 or y>=self.grid.shape[0] or self.grid[y,x] == OFF_TRACK:
                self.y, self.x = self.np_random.choice(np.argwhere(self.grid == START))
                self.vx = 0
                self.vy = 0
                reward = -1
                at_goal = False
                break 
            elif self.grid[y,x] == FINISH:
                reward = 0
                self.x = x
                self.y = y
                at_goal = True
                break
            else: ###SOMETHING WRONG HERE, I MIGHT HAVE ADDED EXTRA CONDITION HERE
                reward = -1
                self.x = new_x
                self.y = new_y
                at_goal = False
            # if new_x >= self.grid_size or new_x < 0 or new_y >= self.grid_size or new_y < 0: #out of range, random re-start, zero velocity
            #     self.y, self.x = self.np_random.choice(np.argwhere(self.grid == START))
            #     self.vx = 0
            #     self.vy = 0
            #     reward = -10
            #     at_goal = False
            # elif self.grid[new_y, new_x] == OFF_TRACK:
            #     self.y, self.x = self.np_random.choice(np.argwhere(self.grid == START))
            #     self.vx = 0
            #     self.vy = 0
            #     reward = -10
            #     at_goal = False
            # elif self.grid[new_y, new_x] == FINISH:
            #     reward = 100
            #     self.x = new_x
            #     self.y = new_y
            #     at_goal = True
            # else:
            #     reward = -1
            #     self.x = new_x
            #     self.y = new_y
            #     at_goal = False
        return StepResult(observation=(self.x, self.y, self.vx, self.vy), reward=reward, terminated=at_goal)


# ---------------------------------------------------------------------- agent
class Jumbo(Agent):
    def __init__(self, observation_space, action_space, epsilon=0.1, gamma=1, seed=None):
        super().__init__(observation_space, action_space, seed)
        # self.q = np.zeros((observation_space.n, action_space.n))
        #self.action_space = action_space #not needed, super init already sets that
        self.policy = np.zeros(self.observation_space.sizes)
        self.q_state_action_value = np.zeros(tuple(self.observation_space.sizes)+(self.action_space.n,))
        self.action_taken = np.zeros(tuple(self.observation_space.sizes)+(self.action_space.n,))
        self.state_action_taken_count = np.zeros(tuple(self.observation_space.sizes)+(self.action_space.n,))

        self.current_episode = []        
        self.epsilon = epsilon
        self.gamma = gamma #discounted reward
        self.episode_truncated = False

    def act(self, obs):
        # ε_t = max( ε_min , ε₀ − (ε₀ − ε_min) · t / T )
        # τ_t = max( τ_min , τ₀ − (τ₀ − τ_min) · t / T )
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_space.n))
        
        best = np.flatnonzero(self.q_state_action_value[tuple(obs)] == self.q_state_action_value[tuple(obs)].max())
        return int(self.rng.choice(best))


    def update(self, t):
        self.current_episode.append((tuple(t.obs), t.action, t.reward))
        self.episode_truncated = t.truncated
        return {}

    def on_episode_start(self, observation) -> None:
        self.current_episode = []
        self.episode_truncated = False

    def on_episode_end(self) -> None:
        if self.episode_truncated: return
        first_appearance = {}
        g = 0
        for index, (obs, action, reward) in enumerate(self.current_episode):
            key = tuple(obs) + (action,)
            if key not in first_appearance:
                first_appearance[key] = index

        for index, (obs, action, reward) in reversed(list(enumerate(self.current_episode))):
            key = tuple(obs) + (action,)
            g = self.gamma * g + reward
            if first_appearance[key] == index:
                self.state_action_taken_count[key] +=1
                self.q_state_action_value[key] += (1/(self.state_action_taken_count[key]))*(g - self.q_state_action_value[key])

    # saving / loading: used by Agent.save(path) and Agent.load(path)
    def state_dict(self):
        return {"q": self.q_state_action_value.copy(),
                "counts": self.state_action_taken_count.copy()}

    def load_state_dict(self, state):
        self.q_state_action_value = state["q"].copy()
        self.state_action_taken_count = state["counts"].copy()


# ----------------------------------------------------------------- experiment

NUM_EPISODES = 10000
NUM_STEPS = 100

def main():
    set_global_seed(0)
    #MY SIMPLE GRID 8X8
    grid_size = 8
    block_size = 5
    grid = np.full((grid_size, grid_size), TRACK, dtype=int)
    grid[:block_size, grid_size - block_size:] = OFF_TRACK     # bottom-right corner blocked
    grid[0, :grid_size - block_size] = START                   # bottom row, left of the block
    grid[block_size:, grid_size - 1] = FINISH                  # right column, above the block

    grid = make_grid(TRACK_3)
    env = RaceTrack(grid=grid, grid_size=grid_size, observation_space=(grid.shape[1], grid.shape[0],5,5), max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, seed=0)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=0.1, gamma= 1, seed=0)

    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=100)])
    history = runner.train(num_episodes=NUM_EPISODES)
    print("Evaluation:", runner.evaluate(num_episodes=10))

    # save the learned agent (Q-values and counts)
    os.makedirs("results", exist_ok=True)
    agent.save("homework2/results/hw2_mc_agent1_track3.pkl")
    np.save("homework2/results/hw2_episode_lengths_agent1_track3.npy", [h.length for h in history])

if __name__ == "__main__":
    main()