# Homework 2 – Racetrack (Sutton & Barto, Exercise 5.12)

Each part has one script that trains everything and then opens the plots. Nothing is saved to
disk. Run the commands from the
`cs138_reinforcement_learning/` folder:

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