"""Low level physics helpers for the simulation."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import PHYSICS, PITCH


def clamp_vector(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > max_norm and norm > 0:
        return vec / norm * max_norm
    return vec


def integrate_motion(position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray, dt: float, max_speed: float) -> Tuple[np.ndarray, np.ndarray]:
    velocity = velocity + acceleration * dt
    velocity = clamp_vector(velocity, max_speed)
    position = position + velocity * dt
    position[0] = np.clip(position[0], 0, PITCH.length)
    position[1] = np.clip(position[1], 0, PITCH.width)
    return position, velocity


def apply_ball_physics(position: np.ndarray, velocity: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    position = position + velocity * dt
    velocity = velocity * PHYSICS.ball_drag
    position[0] = np.clip(position[0], 0, PITCH.length)
    position[1] = np.clip(position[1], 0, PITCH.width)
    return position, velocity
