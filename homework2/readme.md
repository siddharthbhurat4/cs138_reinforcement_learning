
## Setup

The experimental setup consists of three main components:

1. **Environment:** Defines the racetrack, vehicle dynamics,
   state transitions, and reward mechanism.
2. **Agent:** Selects actions and learns state-action values
   from its interactions with the environment.
3. **Interface:** Mediates communication between the agent
   and the environment, ensuring that only the necessary
   information is exchanged.

The reward mechanism determines the reward returned by
the environment after the agent performs an action.

### Environment

The environment is implemented using a base `Environment`
class, which defines common methods for interacting with
and resetting an environment.

The main methods include:

- `step()`: Advances the environment after an agent takes
  an action. It updates the environment state and returns
  the reward and additional information, including the
  observation, optimal action, and whether the selected
  action was optimal.
- `reset()`: Resets the environment state at the beginning
  of a new episode.

The actual environment subclasses override the `step()` method to implement their specific transition and reward mechanisms. For this homework, the `RaceTrack` environment maintains the state of the vehicle, including its position and velocity. The environment's `step()` method applies the selected action according to the racetrack rules, updates the vehicle state, and returns the corresponding reward.Only the observation and reward are made available to the agent for learning. Additional information maintained by the environment is not directly used by the agent.

### Agent

The agent is implemented using a base `Agent` class, which defines methods for action selection, learning, and resetting the agent's internal state.

The main methods include:

- `act()`: Selects an action based on the current observation. During training, the agent uses an epsilon-greedy action-selection strategy. During evaluation, it selects actions greedily.
- `update()`: Collects observations and rewards received during an episode.
- `on_episode_end()`: Processes the collected episode trajectory to calculate returns and update the state-action value estimates.

The agent has nine possible acceleration actions:

```python
ACTIONS = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
```

Each action specifies an acceleration in the horizontal and vertical directions. During an episode, the agent selects actions and collects the resulting observations and rewards. At the end of the episode, `on_episode_end()` processes the collected trajectory in reverse order to calculate the return associated with each state-action pair. These returns are then used to update the agent's state-action value estimates using Monte Carlo control.

### Interface

An interface mediates interactions between the agent and the environment. Its purpose is to ensure that only the necessary information is exchanged between the two components. The environment maintains information such as the optimal action and optimal reward, but this information is not explicitly provided to the agent for learning. The agent learns from the observations and rewards received through interaction with the environment.

# Homework 2 – Racetrack (Sutton & Barto, Exercise 5.12)

Each part has one script that trains everything and then opens the plots. Run the commands from the
`cs138_reinforcement_learning/` folder:

```bash
cd /cs138_reinforcement_learning
```

```bash
pip install -r requirements.txt
```

## Part 1 – single car

Trains on the 8×8 grid, track 1 and track 2 with γ = 0.3, 0.6 and 1.0, then shows the learning
curves and the greedy routes for each track.

```bash
python -m homework2.run_part1
```

## Part 2 – two cars

For track 1 and track 2, first trains car 1 with the Part 1 method, then trains car 2 next to it
with γ = 0.3, 0.6 and 0.9, and shows the learning curves of car 2 and a greedy episode of both cars.

```bash
python -m homework2.run_part2
```

## Part 3 – windy staircase

Trains on the staircase track with wind probability 0.25, 0.5 and 1.0, then shows the greedy
routes and the learning curves.

```bash
python -m homework2.run_part3
```

Training settings (episodes, step limit, γ values, wind probabilities) are constants at the top of
each script.