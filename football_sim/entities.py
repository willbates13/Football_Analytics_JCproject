"""Domain entities used across the simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .constants import PHYSICS, PITCH


@dataclass
class PlayerObservation:
    position: np.ndarray
    velocity: np.ndarray
    has_ball: bool
    stamina: float


@dataclass
class Player:
    identifier: str
    team: str
    max_speed: float = PHYSICS.player_max_speed
    max_acceleration: float = PHYSICS.player_max_acc
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    stamina: float = 1.0
    role: str = "generic"

    def observe(self, ball: "Ball", teammates: List["Player"], opponents: List["Player"]) -> PlayerObservation:
        return PlayerObservation(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            has_ball=ball.owner == self.identifier,
            stamina=self.stamina,
        )


@dataclass
class Ball:
    position: np.ndarray = field(default_factory=lambda: np.array([PITCH.length / 2, PITCH.width / 2]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    owner: Optional[str] = None


@dataclass
class Team:
    name: str
    color: str
    players: List[Player]


@dataclass
class SimulationStats:
    score: Dict[str, int]
    events: List[str] = field(default_factory=list)

    def log_event(self, event: str) -> None:
        self.events.append(event)


@dataclass
class SimulationState:
    time: float
    players: Dict[str, Player]
    ball: Ball
    stats: SimulationStats

    def copy_positions(self) -> Dict[str, np.ndarray]:
        return {pid: player.position.copy() for pid, player in self.players.items()}
