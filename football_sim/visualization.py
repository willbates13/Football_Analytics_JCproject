"""Minimal Matplotlib visualisation for the simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .constants import PITCH
from .entities import SimulationState


@dataclass
class VisualConfig:
    tail_length: int = 20


def animate(states: Iterable[SimulationState], config: VisualConfig | None = None) -> None:
    config = config or VisualConfig()
    states_list = list(states)
    if not states_list:
        raise ValueError("No states were provided for visualisation")

    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_pitch(ax)
    home_scatter = ax.scatter([], [], c="#1565C0", label="Home")
    away_scatter = ax.scatter([], [], c="#E53935", label="Away")
    ball_plot, = ax.plot([], [], "ko", markersize=6)
    plt.legend(loc="upper right")

    def update(frame: int):
        state = states_list[frame]
        home_positions = np.array([player.position for player in state.players.values() if player.team == "home"])
        away_positions = np.array([player.position for player in state.players.values() if player.team == "away"])
        if home_positions.size:
            home_scatter.set_offsets(home_positions)
        if away_positions.size:
            away_scatter.set_offsets(away_positions)
        ball_plot.set_data(state.ball.position[0], state.ball.position[1])
        ax.set_title(f"t = {state.time:.1f}s | Score {state.stats.score['home']} - {state.stats.score['away']}")
        return home_scatter, away_scatter, ball_plot

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(fig, update, frames=len(states_list), interval=50, blit=False, repeat=False)
    plt.show()


def _draw_pitch(ax) -> None:
    ax.set_xlim(0, PITCH.length)
    ax.set_ylim(0, PITCH.width)
    ax.set_aspect('equal')
    ax.add_patch(patches.Rectangle((0, 0), PITCH.length, PITCH.width, fill=False, lw=2))
    ax.axvline(PITCH.length / 2, color='k', linestyle='--', linewidth=1)
    center_circle = patches.Circle((PITCH.length / 2, PITCH.width / 2), 9.15, fill=False, lw=1)
    ax.add_patch(center_circle)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title('Football Simulation')
