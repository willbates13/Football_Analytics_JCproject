"""Global constants for the football simulation engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PitchDimensions:
    """Represents the rectangular playing surface."""

    length: float = 105.0  # meters
    width: float = 68.0


@dataclass(frozen=True)
class GoalDimensions:
    width: float = 7.32


@dataclass(frozen=True)
class SimulationTiming:
    dt: float = 0.1  # seconds per step
    match_duration: float = 90 * 60  # seconds


@dataclass(frozen=True)
class PhysicsConfig:
    player_max_speed: float = 7.0  # m/s
    player_max_acc: float = 3.5  # m/s^2
    ball_drag: float = 0.985
    ball_kick_speed: float = 25.0
    possession_distance: float = 0.8


PITCH = PitchDimensions()
GOAL = GoalDimensions()
TIMING = SimulationTiming()
PHYSICS = PhysicsConfig()
