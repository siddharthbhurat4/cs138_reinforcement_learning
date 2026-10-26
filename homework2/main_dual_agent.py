"""Homework
    python -m homework2.main_dual_agent          # run from the reinforcement_learning/ folder
"""
import os

import numpy as np

from rl.callbacks import ProgressPrinter
from rl.core import Agent, Discrete, MultiDiscrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed
import pickle

from homework2.racetracks import TRACK_1, TRACK_2, make_grid


ACTIONS = [(0,0), (0,1), (0,-1),
           (1,0), (1,1), (1,-1),
           (-1,0), (-1,1), (-1,-1)]

TRACK, OFF_TRACK, START, FINISH = 0, 1, 2, 3
# ---------------------------------------------------------------- environment
class RaceTrack(Environment):

    def __init__(self, grid, grid_size, observation_space, max_episodes, epsilon_vel_noise=0.1, relative_range=3, seed=None):
        self.observation_space = MultiDiscrete(observation_space)   # define spaces first...
        self.action_space = Discrete(len(ACTIONS))

        self.epsilon_vel_noise = epsilon_vel_noise
        self.relative_range = relative_range
        #Grid:
        self.grid_size = grid_size
        # grid[y, x]: x = column (increasing right), y = row (increasing UP, y = 0 is the bottom row)
        #bottom left is 0,0 that is x,y
        self.grid = grid
        self.agent1_q_state_action_value = None
        self.agent1_state_action_taken_count = None
        self.agent1_at_goal = False
        super().__init__(seed=seed, max_episode_steps=max_episodes)  # ...then call super()

    def _reset(self):
        first, second = self.np_random.choice(np.argwhere(self.grid == START), size=2, replace=False)
        self.y1, self.x1 = first
        self.vx1 = 0
        self.vy1 = 0
        self.y2, self.x2 = second
        self.vx2 = 0
        self.vy2 = 0
        self.agent1_at_goal = False
        return self._observation(), {}

    def _observation(self):
        """car 2's own state, plus car 1's position relative to car 2, clipped to +-relative_range and shifted to start at 0."""
        rel_x = int(np.clip(self.x1 - self.x2, -self.relative_range, self.relative_range)) + self.relative_range
        rel_y = int(np.clip(self.y1 - self.y2, -self.relative_range, self.relative_range)) + self.relative_range
        return (int(self.x2), int(self.y2), self.vx2, self.vy2, rel_x, rel_y)
    
    def _start_cell_avoiding(self, x_avoid, y_avoid):
        """excluding the cell (x_avoid, y_avoid)."""
        cells = np.argwhere(self.grid == START)
        cells = cells[~((cells[:, 0] == y_avoid) & (cells[:, 1] == x_avoid))]
        return self.np_random.choice(cells)

    def _step(self, action):
        # car 1 moves first, then car 2
        #AGENT1 STEP
        if not self.agent1_at_goal:
            best = np.flatnonzero(self.agent1_q_state_action_value[tuple((self.x1, self.y1, self.vx1, self.vy1))] == self.agent1_q_state_action_value[tuple((self.x1, self.y1, self.vx1, self.vy1))].max())
            dx1, dy1 = ACTIONS[int(self.np_random.choice(best))]
            #velocity noise:
            if self.np_random.random() < self.epsilon_vel_noise:
                dx1 = 0
                dy1 = 0

            # velocity after this action
            # the other car comes first, and 
            # the learner’s own car comes second. 
            # That works, but the velocity limits use 
            # observation_space.sizes[2] and [3], which are now car 1’s velocity sizes
            new_vx1 = self.vx1 + dx1 if 0 <= self.vx1 + dx1 < self.observation_space.sizes[2] else self.vx1
            new_vy1 = self.vy1 + dy1 if 0 <= self.vy1 + dy1 < self.observation_space.sizes[3] else self.vy1

            # velocity (0, 0) is only allowed on the start line; ignore elsewhere
            on_start_line = self.grid[self.y1, self.x1] == START
            if new_vx1 == 0 and new_vy1 == 0 and not on_start_line:
                new_vx1, new_vy1 = self.vx1, self.vy1

            self.vx1, self.vy1 = new_vx1, new_vy1

            new_x1 = self.x1 + self.vx1
            new_y1 = self.y1 + self.vy1
            points = np.round(np.linspace((self.x1, self.y1), (new_x1, new_y1), max(self.vx1, self.vy1) + 1)).astype(int)
            for x1,y1 in points[1:]:
                if x1<0 or x1>=self.grid.shape[1] or y1<0 or y1>=self.grid.shape[0] or self.grid[y1,x1] == OFF_TRACK:
                    self.y1, self.x1 = self._start_cell_avoiding(self.x2, self.y2)
                    self.vx1 = 0
                    self.vy1 = 0
                    self.agent1_at_goal = False
                    break 
                elif self.grid[y1,x1] == FINISH:
                    self.x1 = x1
                    self.y1 = y1
                    self.vx1 = 0
                    self.vy1 = 0
                    self.agent1_at_goal = True
                    break
                else:
                    self.x1 = x1
                    self.y1 = y1

        #AGENT2 STEP
        dx2, dy2 = ACTIONS[action]
        #velocity noise:
        if self.np_random.random() < self.epsilon_vel_noise:
            dx2 = 0
            dy2 = 0

        # velocity after this action
        new_vx2 = self.vx2 + dx2 if 0 <= self.vx2 + dx2 < self.observation_space.sizes[2] else self.vx2
        new_vy2 = self.vy2 + dy2 if 0 <= self.vy2 + dy2 < self.observation_space.sizes[3] else self.vy2

        # velocity (0, 0) is only allowed on the start line; ignore elsewhere
        on_start_line = self.grid[self.y2, self.x2] == START
        if new_vx2 == 0 and new_vy2 == 0 and not on_start_line:
            new_vx2, new_vy2 = self.vx2, self.vy2

        self.vx2, self.vy2 = new_vx2, new_vy2

        new_x2 = self.x2 + self.vx2
        new_y2 = self.y2 + self.vy2

        if self.vx2 == 0 and self.vy2 == 0:
            #nothing happened and only allowed at start line
            reward = -1
            agent2_at_goal = False

        points = np.round(np.linspace((self.x2, self.y2), (new_x2, new_y2), max(self.vx2, self.vy2) + 1)).astype(int)
        
        for x2,y2 in points[1:]:
            if x2<0 or x2>=self.grid.shape[1] or y2<0 or y2>=self.grid.shape[0] or self.grid[y2,x2] == OFF_TRACK:
                self.y2, self.x2 = self._start_cell_avoiding(self.x1, self.y1)
                self.vx2 = 0
                self.vy2 = 0
                reward = -1
                agent2_at_goal = False
                break 
            elif x2 == self.x1 and y2 == self.y1: #CAR2 COLLIDED WITH CAR1
                self.y2, self.x2 = self._start_cell_avoiding(self.x1, self.y1)
                self.vx2 = 0
                self.vy2 = 0
                reward = -50
                agent2_at_goal = False
                break 
            elif self.grid[y2,x2] == FINISH:
                reward = 0
                self.x2 = x2
                self.y2 = y2
                agent2_at_goal = True
                break
            else:
                reward = -1
                self.x2 = x2
                self.y2 = y2
                agent2_at_goal = False
        return StepResult(observation=self._observation(), reward=reward, terminated=agent2_at_goal)
    
    def load(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.agent1_q_state_action_value = state["q"].copy()
            self.agent1_state_action_taken_count = state["counts"].copy()


# ---------------------------------------------------------------------- agent
class Jumbo(Agent):
    def __init__(self, observation_space, action_space, epsilon=0.1, gamma=1, seed=None):
        super().__init__(observation_space, action_space, seed)
        # self.q = np.zeros((observation_space.n, action_space.n))
        #self.action_space = action_space #not needed, super init already sets that
        # self.policy = np.zeros(self.observation_space.sizes)
        self.q_state_action_value = np.zeros(tuple(self.observation_space.sizes)+(self.action_space.n,))
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
                # self.policy[tuple(obs)] = np.argmax(self.q_state_action_value[tuple(obs)])

    # saving / loading: used by Agent.save(path) and Agent.load(path)
    def state_dict(self):
        return {"q": self.q_state_action_value.copy(),
                "counts": self.state_action_taken_count.copy()}

    def load_state_dict(self, state):
        self.q_state_action_value = state["q"].copy()
        self.state_action_taken_count = state["counts"].copy()


# ----------------------------------------------------------------- experiment

NUM_EPISODES = 15000
NUM_STEPS = 1000
RELATIVE_RANGE = 3
AGENT1_FILE = "homework2/results/hw2_mc_agent1_track2.pkl"

def main():
    set_global_seed(0)
    #MY SIMPLE GRID 8X8
    grid_size = 8
    block_size = 5
    grid = np.full((grid_size, grid_size), TRACK, dtype=int)
    grid[:block_size, grid_size - block_size:] = OFF_TRACK     # bottom-right corner blocked
    grid[0, :grid_size - block_size] = START                   # bottom row, left of the block
    grid[block_size:, grid_size - 1] = FINISH                  # right column, above the block

    grid = make_grid(TRACK_2)
    env = RaceTrack(grid=grid, grid_size=grid_size, observation_space=(grid.shape[1], grid.shape[0],5,5,2*RELATIVE_RANGE+1,2*RELATIVE_RANGE+1), max_episodes=NUM_STEPS, epsilon_vel_noise=0.1, relative_range=RELATIVE_RANGE, seed=0)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=0.1, gamma= 1, seed=0)
    env.load(AGENT1_FILE)
    
    runner = Runner(env, agent, callbacks=[ProgressPrinter(every=50)])
    history = runner.train(num_episodes=NUM_EPISODES)
    print("Evaluation:", runner.evaluate(num_episodes=10))

    # save the learned agent (Q-values and counts)
    os.makedirs("homework2/results", exist_ok=True)
    agent.save("homework2/results/hw2_mc_part2_track2.pkl")
    np.save("homework2/results/hw2_episode_lengths_part2_track2.npy", [h.length for h in history])

if __name__ == "__main__":
    main()