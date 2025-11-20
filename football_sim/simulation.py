"""Top-level simulation environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .constants import GOAL, PHYSICS, PITCH, TIMING
from .controllers import PlayerAction, SimplePolicy
from .entities import Ball, Player, SimulationState, SimulationStats, Team
from .physics import integrate_motion


@dataclass
class SimulationConfig:
    dt: float = TIMING.dt
    duration: float = TIMING.match_duration
    num_players_per_team: int = 5


class FootballSimulation:
    def __init__(self, home_policy: Optional[SimplePolicy] = None, away_policy: Optional[SimplePolicy] = None, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.dt = self.config.dt
        self.time = 0.0
        self.ball = Ball()
        self.home_team = self._create_team("home", "#1565C0", self.config.num_players_per_team, start_x=PITCH.length * 0.25)
        self.away_team = self._create_team("away", "#E53935", self.config.num_players_per_team, start_x=PITCH.length * 0.75)
        self.players: Dict[str, Player] = {p.identifier: p for p in (*self.home_team.players, *self.away_team.players)}
        self.stats = SimulationStats(score={"home": 0, "away": 0})
        self.home_policy = home_policy or SimplePolicy(attacking_goal_x=PITCH.length)
        self.away_policy = away_policy or SimplePolicy(attacking_goal_x=0.0)
        self.history: List[SimulationState] = []

    def reset(self) -> None:
        self.time = 0.0
        self.ball = Ball()
        for player in self.players.values():
            if player.team == "home":
                anchor = np.array([PITCH.length * 0.25, PITCH.width / 2])
            else:
                anchor = np.array([PITCH.length * 0.75, PITCH.width / 2])
            player.position = anchor + np.random.uniform(-5, 5, size=2)
            player.velocity = np.zeros(2)
            player.stamina = 1.0
        self.ball.owner = None
        self.stats = SimulationStats(score={"home": 0, "away": 0})
        self.history.clear()
        self._record_state()

    def run(self) -> SimulationStats:
        steps = int(self.config.duration / self.dt)
        for _ in range(steps):
            self.step()
        return self.stats

    def step(self) -> SimulationState:
        state_view = SimulationState(time=self.time, players=self.players, ball=self.ball, stats=self.stats)
        actions = {}
        actions.update(self.home_policy.select_actions(state_view, [p.identifier for p in self.home_team.players]))
        actions.update(self.away_policy.select_actions(state_view, [p.identifier for p in self.away_team.players]))
        self._apply_player_actions(actions)
        self._resolve_possession()
        self._update_ball()
        self._detect_goals()
        self.time += self.dt
        return self._record_state()

    def _apply_player_actions(self, actions: Dict[str, PlayerAction]) -> None:
        for pid, player in self.players.items():
            action = actions.get(pid, PlayerAction.idle())
            new_pos, new_vel = integrate_motion(
                player.position,
                player.velocity,
                action.acceleration,
                dt=self.dt,
                max_speed=player.max_speed,
            )
            player.position = new_pos
            player.velocity = new_vel

    def _update_ball(self) -> None:
        if self.ball.owner:
            owner = self.players[self.ball.owner]
            self.ball.position = owner.position.copy()
            self.ball.velocity = owner.velocity.copy()
        else:
            self.ball.velocity = np.zeros(2)

    def _resolve_possession(self) -> None:
        closest_player: Optional[Player] = None
        closest_distance = PHYSICS.possession_distance
        for player in self.players.values():
            distance = np.linalg.norm(player.position - self.ball.position)
            if distance <= closest_distance:
                closest_distance = distance
                closest_player = player
        if closest_player:
            self.ball.owner = closest_player.identifier
        elif self.ball.owner:
            self.ball.owner = None

    def _detect_goals(self) -> None:
        y = self.ball.position[1]
        in_goal = abs(y - PITCH.width / 2) < GOAL.width / 2
        if not in_goal:
            return
        if self.ball.position[0] <= 0.5:
            self.stats.score["away"] += 1
            self.stats.log_event(f"{self.time:.1f}: Away goal!")
            self._kickoff(team="home")
        elif self.ball.position[0] >= PITCH.length - 0.5:
            self.stats.score["home"] += 1
            self.stats.log_event(f"{self.time:.1f}: Home goal!")
            self._kickoff(team="away")

    def _kickoff(self, team: str) -> None:
        self.ball.position = np.array([PITCH.length / 2, PITCH.width / 2])
        self.ball.velocity = np.zeros(2)
        self.ball.owner = None

    def _create_team(self, name: str, color: str, num_players: int, start_x: float) -> Team:
        players: List[Player] = []
        for i in range(num_players):
            position = np.array([start_x, (i + 1) / (num_players + 1) * PITCH.width])
            players.append(Player(identifier=f"{name}_{i}", team=name, position=position))
        return Team(name=name, color=color, players=players)

    def _record_state(self) -> SimulationState:
        current = SimulationState(time=self.time, players=self.players, ball=self.ball, stats=self.stats)
        snapshot = self._snapshot(current)
        self.history.append(snapshot)
        return snapshot

    def _snapshot(self, state: SimulationState) -> SimulationState:
        players_copy: Dict[str, Player] = {}
        for pid, player in state.players.items():
            players_copy[pid] = Player(
                identifier=player.identifier,
                team=player.team,
                max_speed=player.max_speed,
                max_acceleration=player.max_acceleration,
                position=player.position.copy(),
                velocity=player.velocity.copy(),
                stamina=player.stamina,
                role=player.role,
            )
        ball_copy = Ball(position=state.ball.position.copy(), velocity=state.ball.velocity.copy(), owner=state.ball.owner)
        stats_copy = SimulationStats(score=state.stats.score.copy(), events=list(state.stats.events))
        return SimulationState(time=state.time, players=players_copy, ball=ball_copy, stats=stats_copy)
