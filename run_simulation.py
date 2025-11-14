"""Command line entry point for the football simulation."""
from __future__ import annotations

import argparse
from importlib import import_module

from football_sim.simulation import FootballSimulation, SimulationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a toy football simulation")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration of the simulated match in seconds")
    parser.add_argument("--dt", type=float, default=0.1, help="Timestep for the integrator")
    parser.add_argument("--players", type=int, default=5, help="Players per team")
    parser.add_argument("--visualize", action="store_true", help="Play back an animation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(dt=args.dt, duration=args.duration, num_players_per_team=args.players)
    sim = FootballSimulation(config=config)
    sim.reset()
    sim.run()
    if args.visualize:
        viz = import_module("football_sim.visualization")
        viz.animate(sim.history)
    else:
        print("Final score:", sim.stats.score)
        print("Events:")
        for event in sim.stats.events:
            print(" -", event)


if __name__ == "__main__":
    main()
