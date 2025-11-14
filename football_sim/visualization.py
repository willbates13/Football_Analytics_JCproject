"""Minimal Matplotlib visualisation for the simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .constants import PITCH, GOAL
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
    home_scatter = ax.scatter([], [], c="#1565C0", edgecolors="white", s=80, label="Home", zorder=3)
    away_scatter = ax.scatter([], [], c="#E53935", edgecolors="white", s=80, label="Away", zorder=3)
    ball_plot, = ax.plot([], [], "o", color="#FFEB3B", markeredgecolor="black", markersize=8, zorder=4)
    plt.legend(loc="upper right")

    def update(frame: int):
        state = states_list[frame]
        home_positions = np.array([player.position for player in state.players.values() if player.team == "home"])
        away_positions = np.array([player.position for player in state.players.values() if player.team == "away"])
        if home_positions.size:
            home_scatter.set_offsets(home_positions)
        else:
            home_scatter.set_offsets(np.empty((0, 2)))
        if away_positions.size:
            away_scatter.set_offsets(away_positions)
        else:
            away_scatter.set_offsets(np.empty((0, 2)))
        ball_plot.set_data([state.ball.position[0]], [state.ball.position[1]])
        ax.set_title(f"t = {state.time:.1f}s | Score {state.stats.score['home']} - {state.stats.score['away']}")
        return home_scatter, away_scatter, ball_plot

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(fig, update, frames=len(states_list), interval=50, blit=False, repeat=False)
    plt.show()


def _draw_pitch(ax) -> None:
    goal_depth = 2.0
    ax.set_xlim(-goal_depth, PITCH.length + goal_depth)
    ax.set_ylim(0, PITCH.width)
    ax.set_aspect('equal')
    _draw_pitch_surface(ax)
    boundary = patches.Rectangle((0, 0), PITCH.length, PITCH.width, fill=False, edgecolor='white', linewidth=2, zorder=2)
    ax.add_patch(boundary)
    ax.axvline(PITCH.length / 2, color='white', linestyle='--', linewidth=1.2, zorder=2)
    center_circle = patches.Circle((PITCH.length / 2, PITCH.width / 2), 9.15, fill=False, linewidth=1.2, edgecolor='white', zorder=2)
    ax.add_patch(center_circle)
    center_spot = patches.Circle((PITCH.length / 2, PITCH.width / 2), 0.3, color='white', zorder=2)
    ax.add_patch(center_spot)

    penalty_box_length = 16.5
    penalty_box_width = 40.32
    six_yard_box_length = 5.5
    six_yard_box_width = 18.32
    for x in (0, PITCH.length - penalty_box_length):
        ax.add_patch(
            patches.Rectangle(
                (x, (PITCH.width - penalty_box_width) / 2),
                penalty_box_length,
                penalty_box_width,
                fill=False,
                edgecolor='white',
                linewidth=1.2,
                zorder=2,
            )
        )
    for x in (0, PITCH.length - six_yard_box_length):
        ax.add_patch(
            patches.Rectangle(
                (x, (PITCH.width - six_yard_box_width) / 2),
                six_yard_box_length,
                six_yard_box_width,
                fill=False,
                edgecolor='white',
                linewidth=1.2,
                zorder=2,
            )
        )

    goal_y = (PITCH.width - GOAL.width) / 2
    ax.add_patch(patches.Rectangle((-goal_depth, goal_y), goal_depth, GOAL.width, fill=False, edgecolor='white', linewidth=1.5, zorder=2))
    ax.add_patch(patches.Rectangle((PITCH.length, goal_y), goal_depth, GOAL.width, fill=False, edgecolor='white', linewidth=1.5, zorder=2))

    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title('Football Simulation')


def _draw_pitch_surface(ax) -> None:
    ax.set_facecolor('#0B5F1A')
    stripe_width = PITCH.length / 10
    stripe_colors = ['#0E6B1D', '#0B5F1A']
    for i in range(10):
        ax.add_patch(
            patches.Rectangle(
                (i * stripe_width, 0),
                stripe_width,
                PITCH.width,
                facecolor=stripe_colors[i % 2],
                edgecolor='none',
                zorder=1,
            )
        )
