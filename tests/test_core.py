"""Run before committing changes to rl/:  python -m pytest"""
from homework_template.main import Corridor, QLearning
from rl.core import Discrete
from rl.runner import Runner


def test_discrete_space():
    s = Discrete(3, seed=0)
    assert all(s.contains(s.sample()) for _ in range(20))
    assert not s.contains(3)


def test_time_limit_truncates():
    env = Corridor(n=50, seed=0)
    agent = QLearning(env.observation_space, env.action_space, epsilon=1.0, seed=0)
    stats = Runner(env, agent).run_episode(train=False)
    assert stats.length <= env.max_episode_steps


def test_training_loop_learns():
    env = Corridor(n=6, seed=0)
    agent = QLearning(env.observation_space, env.action_space, seed=0)
    runner = Runner(env, agent)
    runner.train(100)
    assert runner.evaluate(5)["success_rate"] == 1.0
