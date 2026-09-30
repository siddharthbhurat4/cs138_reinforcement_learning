# Examples

Run each from the `reinforcement_learning/` folder. Each one builds on the previous.

| Script | Shows |
|---|---|
| `python -m examples.ex1_environment` | Writing an `Environment`; reset/step by hand; terminated vs truncated; seeding |
| `python -m examples.ex2_agents_and_runner` | Writing simple `Agent`s; `Runner.run_episode`, `evaluate`, rendering |
| `python -m examples.ex3_learning_agent --plot` | A learning agent: training, metrics, comparing settings, plotting, save/load |
| `python -m examples.ex4_callbacks` | Custom callbacks: outcome counting, recording, early stopping, checkpoints, CSV logs |
| `python -m examples.ex5_continuous_observations` | `Box` observations and extending an agent through inheritance |
