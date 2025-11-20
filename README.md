# Football Simulation Sandbox

This repository now contains a barebones but fully functioning football (soccer) simulation
sandbox that is built entirely from scratch. The aim is to provide a transparent and easily
extensible environment that can later be used for reinforcement-learning experiments. The
codebase deliberately avoids dependencies on any 360-snapshot tooling; everything from the
match engine to the optional visualisation is implemented in simple, object-oriented Python.

## Key components

```
football_sim/
├── constants.py        # Pitch, physics, and timing values
├── controllers.py      # Heuristic action selection policies
├── entities.py         # Data structures for players, ball, teams, and state
├── physics.py          # Low-level integration helpers
├── simulation.py       # FootballSimulation orchestrates the environment
└── visualization.py    # 2D Matplotlib playback of saved states
run_simulation.py       # CLI entry point for simulations
```

## Running a simulation

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Execute a match:

   ```bash
   python run_simulation.py --duration 120 --dt 0.1 --players 5 --visualize
   ```

   Use `--visualize` to open a simple 2D pitch animation. Omitting the flag prints the
   final score and textual event log instead, which makes headless batch simulations
   extremely fast.

## Design notes

- **Deterministic physics:** Player movement obeys configurable speed and acceleration
  limits while the ball follows a simple drag model. The transition function is fully
  deterministic, making it ideal for RL training loops.
- **Scripted policies:** The current `SimplePolicy` is intentionally dumb – players chase
  the ball, dribble towards goal, and occasionally shoot when close enough. Swap it out
  for learned behaviour as soon as you have an agent ready.
- **State tracking:** Each `SimulationState` snapshot records the full scene, so you can
  serialise data, compute rewards, or replay with the visualiser.

Feel free to fork this layout, add logging hooks, richer physics, or reward shaping logic
to suit your experiments.
