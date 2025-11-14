"""Simple controllers that produce player actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from .constants import GOAL, PHYSICS, PITCH
from .entities import Ball, Player, SimulationState
from .physics import clamp_vector


@dataclass
class PlayerAction:
    acceleration: np.ndarray
    kick: np.ndarray

    @staticmethod
    def idle() -> "PlayerAction":
        return PlayerAction(acceleration=np.zeros(2), kick=np.zeros(2))


class SimplePolicy:
    """Very dumb scripted behaviour suitable as a baseline."""

    def __init__(self, attacking_goal_x: float) -> None:
        self.attacking_goal_x = attacking_goal_x

    def select_actions(self, state: SimulationState, player_ids: Iterable[str]) -> Dict[str, PlayerAction]:
        actions: Dict[str, PlayerAction] = {}
        ball = state.ball
        players = state.players

        if ball.owner:
            owner = players[ball.owner]
            kick_vector = self._attempt_shot(owner, ball)
            if np.linalg.norm(kick_vector) > 0:
                actions[owner.identifier] = PlayerAction(np.zeros(2), kick_vector)
        for pid in player_ids:
            player = players[pid]
            desired_velocity = self._desired_velocity(player, ball)
            acceleration = self._compute_acceleration(player, desired_velocity)
            if pid not in actions:
                actions[pid] = PlayerAction(acceleration=acceleration, kick=np.zeros(2))
        return actions

    def _goal_direction(self, team_name: str) -> np.ndarray:  # noqa: ARG002 - hook for custom policies
        direction = -1.0 if self.attacking_goal_x == 0 else 1.0
        return np.array([direction, 0.0])

    def _desired_velocity(self, player: Player, ball: Ball) -> np.ndarray:
        if player.identifier == ball.owner:
            return self._goal_direction(player.team) * player.max_speed
        direction_to_ball = ball.position - player.position
        distance = np.linalg.norm(direction_to_ball)
        if distance < 5.0:
            return clamp_vector(direction_to_ball, player.max_speed)
        anchor_x = PITCH.length * 0.25 if player.team == "home" else PITCH.length * 0.75
        anchor = np.array([anchor_x, PITCH.width / 2])
        return clamp_vector(anchor - player.position, player.max_speed * 0.5)

    def _compute_acceleration(self, player: Player, desired_velocity: np.ndarray) -> np.ndarray:
        delta_v = desired_velocity - player.velocity
        delta_v = clamp_vector(delta_v, player.max_acceleration)
        return delta_v

    def _attempt_shot(self, player: Player, ball: Ball) -> np.ndarray:
        distance_to_goal = abs(ball.position[0] - self.attacking_goal_x)
        centered = abs(ball.position[1] - PITCH.width / 2) < GOAL.width / 2
        if distance_to_goal < 20 and centered:
            direction = np.array([np.sign(self.attacking_goal_x - ball.position[0]), 0.0])
            return direction * PHYSICS.ball_kick_speed
        return np.zeros(2)
