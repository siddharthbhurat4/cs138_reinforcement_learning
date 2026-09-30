# Reinforcement Learning

A small shared base (`rl/`) that every homework imports and builds on.

```
reinforcement_learning/
├── rl/                     shared base — only generic building blocks
│   ├── core/
│   │   ├── environment.py    Environment base class (reset / step, seeding, time limits)
│   │   ├── agent.py          Agent base class (act / update, train/eval mode, save/load)
│   │   ├── spaces.py         Discrete and Box action/observation spaces
│   │   └── types.py          StepResult and Transition data containers
│   ├── runner.py           the interface: runs the agent–environment loop
│   ├── callbacks.py        hooks into the loop (e.g. ProgressPrinter, StepRecorder)
│   └── utils/              set_global_seed, moving_average, plot_curves
├── tests/                  tests rl/
├── requirements.txt
└── .gitignore
```

## Complete Loop

```
          obs                         Transition(s, a, r, s', done)
  Env ─────────────► Runner ─────────────────────────────► Agent.update()
   ▲                  │  ▲                                        │
   └── env.step(a) ◄──┘  └────────── agent.act(obs) ◄─────────────┘
```

- **Environment**: subclass it, set `observation_space` / `action_space`, then call
  `super().__init__()`. Implement `_reset()` → `(obs, info)` and `_step(action)` → `StepResult`.
- **Agent**: subclass it and implement `act(obs)` (explore when `self.training`) and
  `update(transition)` (learn; return a dict of metrics).
- **Runner**: `Runner(env, agent).train(n)` and `.evaluate(n)`. Neither side knows about the other.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running a homework

```bash
python -m homework1.main        # always run from the reinforcement_learning/ folder
```

<!--
## Before committing changes to `rl/`

```bash
python -m pytest
git tag hw1-submitted           # after each submission, to freeze that version
```
-->
