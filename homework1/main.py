"""Homework 1:

    cp -r homework_template homework1
    python -m homework1.main          # run from the reinforcement_learning/ folder

Everything below is an example to replace with your own env and agent.
As the homework grows, split them into env.py / agent.py in this folder.
"""
import numpy as np

from rl.callbacks import ProgressPrinter, Callback
from rl.core import Agent, Discrete, Environment, StepResult
from rl.runner import Runner
from rl.utils import set_global_seed
from homework1.plotting import plot_bandit_results



#environment
class TenArmedBandit(Environment):
    """start with equal values of q* and then do random walk."""

    def __init__(self, n_action: int = 10, max_steps = 10000, rewards_standarddeviation = 0.05, randomwalk_mean = 0.0, randomwalk_standarddeviation = 0.01, seed=0):
        self.n_observation = 1
        self.n_action = n_action
        self.observation_space = Discrete(self.n_observation)   # define spaces first...
        self.action_space = Discrete(n_action)
        self.reward_means = np.zeros(n_action)
        self.rewards_standarddeviation = rewards_standarddeviation
        self.randomwalk_mean = randomwalk_mean
        self.randomwalk_standarddeviation = randomwalk_standarddeviation
        #we get random numbser generator from our environment base class
        # so we can use that : self.np_random
        super().__init__(seed=seed, max_episode_steps=max_steps)  # ...then call super()

    def _reset(self):
        self.reward_means = np.zeros(self.n_action)
        return 0, {}
    
    def _rewardrandomwalk(self):
        random_movement = self.np_random.normal(loc=self.randomwalk_mean, scale = self.randomwalk_standarddeviation, size = self.n_action)#get a random number array of 10 or or of size action_space
        self.reward_means += random_movement#apply that random movement

    def _step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        #need to decide terminated method: truncation is already being done in base environment step
        thisStepReward = float(self.np_random.normal(loc = self.reward_means[action], scale=self.rewards_standarddeviation))
        thisStepBestAction = int(np.argmax(self.reward_means))
        self._rewardrandomwalk()
        return StepResult(observation=0, reward = thisStepReward, terminated=False, info = {"optimal_action": thisStepBestAction, "is_optimal": bool(thisStepBestAction == action)})
    
    #maybe render and show how my multi armed bandit 
    # is changing with random walk !!
    # def render(self):
    #     return super().render()

#agent
class Jumbo(Agent):
    def __init__(self, observation_space, action_space, alpha=0.1, epsilon=0.1, seed=None, action_value_update_method = "sample_averaging"):
        super().__init__(observation_space, action_space, seed)
        self.alpha, self.epsilon = alpha, epsilon
        # self.q = np.zeros((observation_space.n, action_space.n))
        #self.action_space = action_space #not needed, super init already sets that
        self.action_value = np.zeros(self.action_space.n)
        self.reward_store = [[] for _ in range(self.action_space.n)]
        self.action_taken = np.zeros(self.action_space.n)
        self.action_value_updated_method = action_value_update_method
        self.temperature = 1.0
        self.action_value_exp_over_temp = np.zeros(self.action_space.n)
        self.action_value_softmax = np.zeros(self.action_space.n)
        self.action_value_exp_over_temp_sum = 0

    def act(self, obs):
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_space.n))
        # best = np.flatnonzero(self.action_value == self.action_value.max())

        #softmax action selection method
        #divide by temperature
        self.action_value_exp_over_temp = np.exp(self.action_value/self.temperature)
        self.action_value_exp_over_temp_sum = sum(self.action_value_exp_over_temp)
        self.action_value_softmax = self.action_value_exp_over_temp / self.action_value_exp_over_temp_sum
        best = np.flatnonzero(self.action_value_softmax == self.action_value_softmax.max())

        return int(self.rng.choice(self.action_space.n, p=self.action_value_softmax))  # random tie-breaking

    def update(self, t):
        # Updating action value using sample averaging method that keeps track of all rewards
        if self.action_value_updated_method == "sample_averaging":
            self.reward_store[t.action].append(t.reward)
            self.action_value[t.action] = sum(self.reward_store[t.action])/len(self.reward_store[t.action])
        # Updating action value using incremental average update method 
        # that doesnt need to keep track of all rewards
        # Q_n+1 = Q_n + alpha * (R_n - Q_n)
        elif self.action_value_updated_method == "incremental_average":
            self.action_taken[t.action] +=1
            self.action_value[t.action] += (1/(self.action_taken[t.action]))*(t.reward - self.action_value[t.action])
        #updating action value using step size parameter alpha
        elif self.action_value_updated_method == "step_size_method":
            self.action_value[t.action] += self.alpha * (t.reward - self.action_value[t.action])
        else:
            ValueError()

        return {}

    def on_episode_start(self, observation) -> None:
        self.action_taken = np.zeros(self.action_space.n)
        self.action_value = np.zeros(self.action_space.n)
        self.reward_store = [[] for _ in range(self.action_space.n)]

    def on_episode_end(self) -> None:
        pass

class StepRewardRecorder(Callback):
    """Keeps every per-step reward, e.g. for 'average reward per step' plots."""

    def __init__(self):
        self.episodes = []
        self.optimal_action = []

    def on_episode_start(self, runner, observation):
        self.episodes.append([])
        self.optimal_action.append([])

    def on_step(self, runner, transition, metrics):
        self.episodes[-1].append(transition.reward)
        self.optimal_action[-1].append(transition.info["is_optimal"])



# def main():
#     set_global_seed(0)
#     env = TenArmedBandit(n_action=10, max_steps=10000, rewards_standarddeviation = 0.05, randomwalk_mean = 0.0, randomwalk_standarddeviation = 0.01, seed=0)
#     agent_sample_average = Jumbo(env.observation_space, env.action_space, epsilon=0.1, seed=0, action_value_update_method="sample_averaging")
#     agent_incremental_average = Jumbo(env.observation_space, env.action_space, epsilon=0.1, seed=0, action_value_update_method="incremental_average")
#     agent_step_size = Jumbo(env.observation_space, env.action_space, epsilon=0.1, seed=0, action_value_update_method="step_size_method")
#     stepRecorder = StepRewardRecorder()
#     runner = Runner(env, agent_sample_average, callbacks=[stepRecorder])
#     history = runner.train(num_episodes=1)
#     # print("Evaluation:", runner.evaluate(num_episodes=10))

#     # from rl.utils import plot_curves
#     # plot_curves({"Q-learning": [h.total_reward for h in history]}, window=10)
#     results = {
#     "sample average": (stepRecorder.episodes, stepRecorder.optimal_action),
#     }
#     plot_bandit_results(results, window=100)


# if __name__ == "__main__":
#     main()

# Settings in one place so they're easy to change and to report
N_STEPS = 10_000
N_EPISODES = 1
EPSILON = 0.1
ALPHA = 0.1         # step size for the constant-step-size method
WINDOW = 100        # moving-average window for the plots
SEED = 0
 
METHODS = {        # plot label -> action_value_update_method passed to the agent
    "sample average": "sample_averaging",
    "incremental average": "incremental_average",
    f"constant step size (α={ALPHA})": "step_size_method",
}
 
 
def run_method(method):
    env = TenArmedBandit(n_action=10, max_steps=N_STEPS, rewards_standarddeviation=0.05,
                         randomwalk_mean=0.0, randomwalk_standarddeviation=0.01, seed=SEED)
    agent = Jumbo(env.observation_space, env.action_space, epsilon=EPSILON, alpha=ALPHA,
                  seed=SEED, action_value_update_method=method)
    recorder = StepRewardRecorder()
    Runner(env, agent, callbacks=[recorder]).train(num_episodes=N_EPISODES)
    return recorder.episodes, recorder.optimal_action
 
 
def main():
    set_global_seed(SEED)
    results = {label: run_method(method) for label, method in METHODS.items()}
 
    # Summary table: performance over the last 1,000 steps
    print(f"{'method':32s} {'avg reward':>11s} {'% optimal':>10s}   (last 1000 steps)")
    for label, (rewards, is_optimal) in results.items():
        rewards, is_optimal = np.array(rewards), np.array(is_optimal)
        print(f"{label:32s} {rewards[:, -1000:].mean():11.3f} {100 * is_optimal[:, -1000:].mean():10.1f}")
 
    # Sanity check
    sample = np.array(results["sample average"][0])
    incremental = np.array(results["incremental average"][0])
    print(f"\nmax |reward difference| sample vs incremental: {np.max(np.abs(sample - incremental)):.2e}")
 
    plot_bandit_results(results, window=WINDOW, layout="columns")
 
 
if __name__ == "__main__":
    main()