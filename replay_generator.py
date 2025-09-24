#!/usr/bin/env python3
"""StatsBomb 360 match replay generator.

This script downloads StatsBomb open data (events and 360 freeze frames)
for a single match and builds an interactive Plotly playback that can be
used to visually replay the match phase by phase. The resulting HTML
contains a slider-driven animation of the players on a full pitch,
annotated with event metadata, possession changes, ball carrier details,
and a rich set of shot-related insights.

Alongside the visual output the script caches the raw data locally and
stores tidy JSON representations of the processed events and freeze frame
snapshots, making it easier to re-use the data for future analysis (for
example, Markov decision process modelling).
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    go = None
    make_subplots = None

DATA_REPO_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
GITHUB_API_THREE_SIXTY = (
    "https://api.github.com/repos/statsbomb/open-data/contents/data/three-sixty"
)


@dataclass
class TeamInfo:
    team_id: int
    name: str
    side: str  # "home" or "away"


@dataclass
class PlayerInfo:
    player_id: int
    name: str
    jersey_number: Optional[int]
    team_id: int
    team_name: str
    side: str


@dataclass
class ShotRecord:
    minute: int
    second: int
    x: Optional[float]
    y: Optional[float]
    xg: float
    outcome: str
    goal: bool
    team_side: str


@dataclass
class EventRecord:
    index: int
    event_uuid: str
    minute: int
    second: int
    timestamp: str
    team_id: Optional[int]
    team_name: Optional[str]
    player_id: Optional[int]
    player_name: Optional[str]
    event_type: str
    detail: Optional[str]
    outcome: Optional[str]
    possession_team_id: Optional[int]
    possession_team_name: Optional[str]
    possession_number: Optional[int]
    possession_change: bool
    possession_from: Optional[int]
    possession_to: Optional[int]
    location: Tuple[Optional[float], Optional[float]]
    end_location: Tuple[Optional[float], Optional[float]]
    carry_end_location: Tuple[Optional[float], Optional[float]]
    shot_end_location: Tuple[Optional[float], Optional[float]]
    shot_xg: float
    shot_outcome: Optional[str]
    is_goal: bool
    score_home: int
    score_away: int
    shots_home: int
    shots_away: int
    shots_on_target_home: int
    shots_on_target_away: int
    xg_home: float
    xg_away: float
    period: int
    visible_area: Optional[List[float]]

    @property
    def match_time_minutes(self) -> float:
        return self.minute + self.second / 60.0

    @property
    def match_time_seconds(self) -> float:
        return self.minute * 60.0 + self.second


@dataclass
class FreezeFrameRecord:
    event_uuid: str
    player_id: Optional[int]
    player_name: Optional[str]
    team_id: Optional[int]
    team_name: Optional[str]
    side: Optional[str]
    x: Optional[float]
    y: Optional[float]
    teammate: bool
    actor: bool
    keeper: bool


class DataManager:
    """Download and cache StatsBomb open data."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.raw_dir = self.base_dir / "raw"
        self.matches_dir = self.raw_dir / "matches"
        self.events_dir = self.raw_dir / "events"
        self.lineups_dir = self.raw_dir / "lineups"
        self.three_sixty_dir = self.raw_dir / "three_sixty"
        for directory in (
            self.base_dir,
            self.raw_dir,
            self.matches_dir,
            self.events_dir,
            self.lineups_dir,
            self.three_sixty_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _fetch_json(self, url: str, cache_path: Path) -> List[dict]:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        request = urllib.request.Request(url, headers={"User-Agent": "statsbomb-replay"})
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
        cache_path.write_text(payload, encoding="utf-8")
        return json.loads(payload)

    def get_competitions(self) -> List[dict]:
        url = f"{DATA_REPO_URL}/competitions.json"
        cache_path = self.raw_dir / "competitions.json"
        return self._fetch_json(url, cache_path)

    def get_matches(self, competition_id: int, season_id: int) -> List[dict]:
        url = f"{DATA_REPO_URL}/matches/{competition_id}/{season_id}.json"
        cache_path = self.matches_dir / f"{competition_id}_{season_id}.json"
        return self._fetch_json(url, cache_path)

    def get_events(self, match_id: int) -> List[dict]:
        url = f"{DATA_REPO_URL}/events/{match_id}.json"
        cache_path = self.events_dir / f"{match_id}.json"
        return self._fetch_json(url, cache_path)

    def get_three_sixty(self, match_id: int) -> List[dict]:
        url = f"{DATA_REPO_URL}/three-sixty/{match_id}.json"
        cache_path = self.three_sixty_dir / f"{match_id}.json"
        return self._fetch_json(url, cache_path)

    def get_lineups(self, match_id: int) -> List[dict]:
        url = f"{DATA_REPO_URL}/lineups/{match_id}.json"
        cache_path = self.lineups_dir / f"{match_id}.json"
        return self._fetch_json(url, cache_path)

    def list_three_sixty_match_ids(self) -> List[int]:
        request = urllib.request.Request(
            GITHUB_API_THREE_SIXTY,
            headers={"User-Agent": "statsbomb-replay"},
        )
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - network
            raise RuntimeError(f"Failed to list 360 matches: {exc}") from exc
        listing = json.loads(payload)
        match_ids: List[int] = []
        for item in listing:
            name = item.get("name", "")
            if name.endswith(".json"):
                try:
                    match_ids.append(int(name.split(".")[0]))
                except ValueError:
                    continue
        return match_ids

    def save_processed_dataset(
        self,
        match_id: int,
        events: List[EventRecord],
        freeze_frames: List[FreezeFrameRecord],
    ) -> None:
        processed_dir = self.base_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        events_payload = [event.__dict__ for event in events]
        freeze_payload = [frame.__dict__ for frame in freeze_frames]
        (processed_dir / f"{match_id}_events.json").write_text(
            json.dumps(events_payload, indent=2),
            encoding="utf-8",
        )
        (processed_dir / f"{match_id}_freeze_frames.json").write_text(
            json.dumps(freeze_payload, indent=2),
            encoding="utf-8",
        )


def build_team_directory(match_info: dict) -> Dict[int, TeamInfo]:
    home = match_info.get("home_team") or {}
    away = match_info.get("away_team") or {}
    directory: Dict[int, TeamInfo] = {}
    if "home_team_id" in home:
        directory[int(home["home_team_id"])] = TeamInfo(
            team_id=int(home["home_team_id"]),
            name=home.get("home_team_name"),
            side="home",
        )
    if "away_team_id" in away:
        directory[int(away["away_team_id"])] = TeamInfo(
            team_id=int(away["away_team_id"]),
            name=away.get("away_team_name"),
            side="away",
        )
    return directory


def build_player_directory(lineups: List[dict], teams: Dict[int, TeamInfo]) -> Dict[int, PlayerInfo]:
    directory: Dict[int, PlayerInfo] = {}
    for team_entry in lineups:
        team_id = team_entry.get("team_id")
        team_info = teams.get(team_id)
        for player in team_entry.get("lineup", []):
            player_id = player.get("player_id")
            directory[player_id] = PlayerInfo(
                player_id=player_id,
                name=player.get("player_name"),
                jersey_number=player.get("jersey_number"),
                team_id=team_id,
                team_name=team_info.name if team_info else None,
                side=team_info.side if team_info else None,
            )
    return directory


def normalise_location(value: Optional[Iterable[Optional[float]]]) -> Tuple[Optional[float], Optional[float]]:
    if not value:
        return (None, None)
    coords = list(value)
    if len(coords) < 2:
        coords += [None] * (2 - len(coords))
    return (coords[0], coords[1])


def compute_event_detail(event: dict) -> Tuple[Optional[str], Optional[str]]:
    event_type = event.get("type", {}).get("name")
    detail = None
    outcome = None
    if event_type == "Pass":
        pass_info = event.get("pass", {})
        recipient = pass_info.get("recipient", {}).get("name")
        height = pass_info.get("height", {}).get("name")
        detail = ", ".join(
            [
                part
                for part in (
                    f"to {recipient}" if recipient else None,
                    pass_info.get("body_part", {}).get("name"),
                    height,
                )
                if part
            ]
        ) or None
        outcome = pass_info.get("outcome", {}).get("name")
    elif event_type == "Carry":
        carry_info = event.get("carry", {})
        detail = (
            f"to {carry_info.get('end_location')}" if carry_info.get("end_location") else None
        )
    elif event_type == "Dribble":
        detail = event.get("dribble", {}).get("outcome", {}).get("name")
    elif event_type == "Shot":
        shot_info = event.get("shot", {})
        outcome = shot_info.get("outcome", {}).get("name")
        technique = shot_info.get("technique", {}).get("name")
        body = shot_info.get("body_part", {}).get("name")
        detail = ", ".join([part for part in (technique, body) if part]) or None
    elif event_type == "Pressure":
        detail = "Counterpress" if event.get("counterpress") else None
    elif event_type == "Ball Receipt*":
        outcome = event.get("ball_receipt", {}).get("outcome", {}).get("name")
    elif event_type == "Foul Won":
        detail = "Defensive" if event.get("foul_won", {}).get("defensive") else None
    elif event_type == "Foul Committed":
        detail = event.get("foul_committed", {}).get("type", {}).get("name")
    elif event_type == "Substitution":
        sub = event.get("substitution", {})
        detail = sub.get("replacement", {}).get("name")
    return detail, outcome


def process_events(
    raw_events: List[dict],
    freeze_by_event: Dict[str, dict],
    players: Dict[int, PlayerInfo],
    teams: Dict[int, TeamInfo],
) -> Tuple[List[EventRecord], List[FreezeFrameRecord], List[ShotRecord]]:
    events: List[EventRecord] = []
    freeze_records: List[FreezeFrameRecord] = []
    shots: List[ShotRecord] = []

    home_team_id = next(team.team_id for team in teams.values() if team.side == "home")
    away_team_id = next(team.team_id for team in teams.values() if team.side == "away")

    score_home = 0
    score_away = 0
    shots_home = 0
    shots_away = 0
    shots_on_target_home = 0
    shots_on_target_away = 0
    xg_home = 0.0
    xg_away = 0.0
    previous_possession_team: Optional[int] = None

    for index, event in enumerate(raw_events):
        event_uuid = event.get("id")
        minute = int(event.get("minute", 0))
        second = int(event.get("second", 0))
        team_id = event.get("team", {}).get("id")
        team_info = teams.get(team_id)
        team_name = team_info.name if team_info else event.get("team", {}).get("name")
        player_id = event.get("player", {}).get("id")
        player_info = players.get(player_id)
        player_name = player_info.name if player_info else event.get("player", {}).get("name")
        event_type = event.get("type", {}).get("name") or "Unknown"
        detail, outcome = compute_event_detail(event)

        location = normalise_location(event.get("location"))
        end_location = normalise_location(event.get("pass", {}).get("end_location"))
        carry_end_location = normalise_location(event.get("carry", {}).get("end_location"))
        shot_end_location = normalise_location(event.get("shot", {}).get("end_location"))

        shot_info = event.get("shot")
        shot_xg = float(shot_info.get("statsbomb_xg", 0.0)) if shot_info else 0.0
        shot_outcome = shot_info.get("outcome", {}).get("name") if shot_info else None
        is_goal = bool(shot_info and shot_outcome == "Goal")
        if shot_info:
            if team_id == home_team_id:
                shots_home += 1
                xg_home += shot_xg
                if shot_outcome in {"Saved", "Goal", "ShotOnPost", "SavedToPost"}:
                    shots_on_target_home += 1
                if is_goal:
                    score_home += 1
            elif team_id == away_team_id:
                shots_away += 1
                xg_away += shot_xg
                if shot_outcome in {"Saved", "Goal", "ShotOnPost", "SavedToPost"}:
                    shots_on_target_away += 1
                if is_goal:
                    score_away += 1
            shots.append(
                ShotRecord(
                    minute=minute,
                    second=second,
                    x=location[0],
                    y=location[1],
                    xg=shot_xg,
                    outcome=shot_outcome or "Unknown",
                    goal=is_goal,
                    team_side=team_info.side if team_info else "",
                )
            )

        possession_team_id = event.get("possession_team", {}).get("id")
        possession_team_name = event.get("possession_team", {}).get("name")
        possession_change = False
        possession_from = previous_possession_team
        possession_to = possession_team_id
        if previous_possession_team != possession_team_id:
            possession_change = previous_possession_team is not None
            previous_possession_team = possession_team_id

        record = EventRecord(
            index=index,
            event_uuid=event_uuid,
            minute=minute,
            second=second,
            timestamp=event.get("timestamp"),
            team_id=team_id,
            team_name=team_name,
            player_id=player_id,
            player_name=player_name,
            event_type=event_type,
            detail=detail,
            outcome=outcome,
            possession_team_id=possession_team_id,
            possession_team_name=possession_team_name,
            possession_number=event.get("possession"),
            possession_change=possession_change,
            possession_from=possession_from,
            possession_to=possession_to,
            location=location,
            end_location=end_location,
            carry_end_location=carry_end_location,
            shot_end_location=shot_end_location,
            shot_xg=shot_xg,
            shot_outcome=shot_outcome,
            is_goal=is_goal,
            score_home=score_home,
            score_away=score_away,
            shots_home=shots_home,
            shots_away=shots_away,
            shots_on_target_home=shots_on_target_home,
            shots_on_target_away=shots_on_target_away,
            xg_home=xg_home,
            xg_away=xg_away,
            period=int(event.get("period", 1)),
            visible_area=freeze_by_event.get(event_uuid, {}).get("visible_area"),
        )
        events.append(record)

        freeze_data = freeze_by_event.get(event_uuid, {})
        for player in freeze_data.get("freeze_frame", []) or []:
            player_entry = player.get("player") or {}
            player_id_ff = player_entry.get("id")
            info = players.get(player_id_ff)
            freeze_records.append(
                FreezeFrameRecord(
                    event_uuid=event_uuid,
                    player_id=player_id_ff,
                    player_name=player_entry.get("name"),
                    team_id=info.team_id if info else None,
                    team_name=info.team_name if info else None,
                    side=info.side if info else None,
                    x=player.get("location", [None, None])[0]
                    if player.get("location")
                    else None,
                    y=player.get("location", [None, None])[1]
                    if player.get("location")
                    else None,
                    teammate=bool(player.get("teammate")),
                    actor=bool(player.get("actor")),
                    keeper=bool(player.get("keeper")),
                )
            )

    return events, freeze_records, shots


def freeze_frames_by_event(freeze_data: List[dict]) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for entry in freeze_data:
        event_uuid = entry.get("event_uuid")
        if event_uuid:
            mapping[event_uuid] = entry
    return mapping


def build_pitch(fig: go.Figure, row: int, col: int, total_cols: int = 2) -> None:
    axis_index = (row - 1) * total_cols + col
    scaleanchor = "x" if axis_index == 1 else f"x{axis_index}"
    fig.update_xaxes(range=[0, 120], showgrid=False, zeroline=False, visible=False, row=row, col=col)
    fig.update_yaxes(
        range=[-5, 85],
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor=scaleanchor,
        scaleratio=1,
        row=row,
        col=col,
    )
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=120,
        y1=80,
        line=dict(color="white", width=2),
        row=row,
        col=col,
    )
    fig.add_shape(
        type="line",
        x0=60,
        y0=0,
        x1=60,
        y1=80,
        line=dict(color="white", width=2),
        row=row,
        col=col,
    )
    fig.add_shape(
        type="circle",
        x0=60 - 9.15,
        y0=40 - 9.15,
        x1=60 + 9.15,
        y1=40 + 9.15,
        line=dict(color="white", width=2),
        row=row,
        col=col,
    )
    fig.add_shape(
        type="circle",
        x0=60 - 0.5,
        y0=40 - 0.5,
        x1=60 + 0.5,
        y1=40 + 0.5,
        fillcolor="white",
        line=dict(color="white", width=1),
        row=row,
        col=col,
    )
    for side_start in (0, 120):
        penalty_x0 = 0 if side_start == 0 else 120 - 18
        penalty_x1 = 18 if side_start == 0 else 120
        six_x0 = 0 if side_start == 0 else 120 - 6
        six_x1 = 6 if side_start == 0 else 120
        spot_x = 11 if side_start == 0 else 120 - 11
        fig.add_shape(
            type="rect",
            x0=penalty_x0,
            y0=18,
            x1=penalty_x1,
            y1=62,
            line=dict(color="white", width=2),
            row=row,
            col=col,
        )
        fig.add_shape(
            type="rect",
            x0=six_x0,
            y0=30,
            x1=six_x1,
            y1=50,
            line=dict(color="white", width=2),
            row=row,
            col=col,
        )
        fig.add_shape(
            type="circle",
            x0=spot_x - 0.5,
            y0=40 - 0.5,
            x1=spot_x + 0.5,
            y1=40 + 0.5,
            fillcolor="white",
            line=dict(color="white", width=1),
            row=row,
            col=col,
        )


def timeline_marker_style(event: EventRecord) -> Tuple[str, float, str]:
    if event.is_goal:
        return "#ffd700", 16.0, "star"
    if event.event_type == "Shot":
        return "#ff7f0e", 12.0, "triangle-up"
    if event.possession_change:
        return "#f0f0f0", 12.0, "x"
    if event.event_type in {"Foul Won", "Foul Committed"}:
        return "#bcbd22", 10.0, "square"
    if event.event_type in {"Pressure", "Duel", "Dribble"}:
        return "#17becf", 10.0, "diamond"
    return "#1f77b4", 8.0, "circle"


def build_event_summary(event: EventRecord, teams: Dict[int, TeamInfo]) -> str:
    time_str = f"{event.minute:02d}:{event.second:02d}"
    team_name = event.team_name or (
        teams[event.team_id].name if event.team_id in teams else "Unknown"
    )
    player_name = event.player_name or "Unknown"
    detail = f" ({event.detail})" if event.detail else ""
    shot_extension = ""
    if event.event_type == "Shot":
        shot_extension = f" - xG {event.shot_xg:.2f} ({event.shot_outcome})"
    possession_change = ""
    if event.possession_change:
        from_name = teams[event.possession_from].name if event.possession_from in teams else "Unknown"
        to_name = teams[event.possession_to].name if event.possession_to in teams else "Unknown"
        possession_change = f"\nPossession: {from_name} ➜ {to_name}"
    goal_marker = " ⚽" if event.is_goal else ""
    return (
        f"{time_str} — {team_name} — {player_name}\n"
        f"{event.event_type}{detail}{shot_extension}{goal_marker}{possession_change}"
    )


def build_scoreboard_text(event: EventRecord, teams: Dict[int, TeamInfo]) -> str:
    home_team = next(team for team in teams.values() if team.side == "home")
    away_team = next(team for team in teams.values() if team.side == "away")
    possession_name = event.possession_team_name or (
        teams[event.possession_team_id].name if event.possession_team_id in teams else "N/A"
    )
    lines = [
        f"<b>{home_team.name}</b> {event.score_home} - {event.score_away} <b>{away_team.name}</b>",
        f"Possession: {possession_name}",
        f"Shots (on target): {event.shots_home} ({event.shots_on_target_home}) | {event.shots_away} ({event.shots_on_target_away})",
        f"xG: {event.xg_home:.2f} | {event.xg_away:.2f}",
    ]
    if event.event_type == "Shot":
        lines.append(
            f"Latest shot: {event.player_name or 'Unknown'} — xG {event.shot_xg:.2f} ({event.shot_outcome})"
        )
    return "<br>".join(lines)


def split_freeze_players(
    freeze_entry: List[FreezeFrameRecord],
    home_color: str,
    away_color: str,
) -> Tuple[go.Scatter, go.Scatter]:
    home_x: List[Optional[float]] = []
    home_y: List[Optional[float]] = []
    home_text: List[str] = []
    home_sizes: List[float] = []
    home_line_colors: List[str] = []
    home_line_widths: List[float] = []

    away_x: List[Optional[float]] = []
    away_y: List[Optional[float]] = []
    away_text: List[str] = []
    away_sizes: List[float] = []
    away_line_colors: List[str] = []
    away_line_widths: List[float] = []

    for player in freeze_entry:
        text = player.player_name or "Unknown"
        if player.side == "home":
            home_x.append(player.x)
            home_y.append(player.y)
            label = text
            if player.keeper:
                label = f"{label} (GK)"
            home_text.append(label)
            size = 22 if player.actor else 16
            line_color = "#ffd700" if player.actor else "white"
            home_sizes.append(size)
            home_line_colors.append(line_color)
            home_line_widths.append(3 if player.actor else 1.5)
        elif player.side == "away":
            away_x.append(player.x)
            away_y.append(player.y)
            label = text
            if player.keeper:
                label = f"{label} (GK)"
            away_text.append(label)
            size = 22 if player.actor else 16
            line_color = "#ffd700" if player.actor else "white"
            away_sizes.append(size)
            away_line_colors.append(line_color)
            away_line_widths.append(3 if player.actor else 1.5)

    home_trace = go.Scatter(
        x=home_x,
        y=home_y,
        mode="markers",
        marker=dict(
            color=home_color,
            size=home_sizes,
            line=dict(color=home_line_colors, width=home_line_widths),
        ),
        text=home_text,
        name="Home team",
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    away_trace = go.Scatter(
        x=away_x,
        y=away_y,
        mode="markers",
        marker=dict(
            color=away_color,
            size=away_sizes,
            line=dict(color=away_line_colors, width=away_line_widths),
        ),
        text=away_text,
        name="Away team",
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    return home_trace, away_trace


def infer_ball_position(event: EventRecord, freeze_entry: List[FreezeFrameRecord]) -> Tuple[Optional[float], Optional[float]]:
    if event.location != (None, None):
        return event.location
    for player in freeze_entry:
        if player.actor and player.x is not None and player.y is not None:
            return (player.x, player.y)
    return (None, None)


def build_replay_figure(
    events: List[EventRecord],
    freeze_frames: List[FreezeFrameRecord],
    shots: List[ShotRecord],
    teams: Dict[int, TeamInfo],
    match_info: dict,
) -> go.Figure:
    if go is None or make_subplots is None:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "Plotly is required to build the replay figure. Install it with 'pip install plotly'."
        )

    freeze_lookup: Dict[str, List[FreezeFrameRecord]] = {}
    for frame in freeze_frames:
        freeze_lookup.setdefault(frame.event_uuid, []).append(frame)

    events_with_frames = [event for event in events if event.event_uuid in freeze_lookup]
    if not events_with_frames:
        raise ValueError("No events with 360 freeze frames available for this match.")

    home_color = "#1f77b4"
    away_color = "#d62728"

    timeline_x = [event.match_time_minutes for event in events_with_frames]
    timeline_y = [0 for _ in events_with_frames]
    timeline_colors = []
    timeline_sizes = []
    timeline_symbols = []
    for event in events_with_frames:
        color, size, symbol = timeline_marker_style(event)
        timeline_colors.append(color)
        timeline_sizes.append(size)
        timeline_symbols.append(symbol)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter"}], [{"type": "scatter"}, {"type": "scatter"}]],
        column_widths=[0.7, 0.3],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=("Pitch Replay", "Match Centre", "Event Timeline", "Shot Map"),
    )

    build_pitch(fig, row=1, col=1)
    build_pitch(fig, row=2, col=2)

    fig.add_trace(
        go.Scatter(
            x=timeline_x,
            y=timeline_y,
            mode="markers",
            marker=dict(color=timeline_colors, size=timeline_sizes, symbol=timeline_symbols),
            text=[build_event_summary(event, teams) for event in events_with_frames],
            hovertemplate="%{text}<extra></extra>",
            name="Timeline events",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[-1, 1], visible=False, row=2, col=1)
    max_minute = max(event.match_time_minutes for event in events_with_frames)
    fig.update_xaxes(range=[-1, max_minute + 1], row=2, col=1, title="Minute", showgrid=False)

    home_shots = [shot for shot in shots if shot.team_side == "home"]
    away_shots = [shot for shot in shots if shot.team_side == "away"]
    fig.add_trace(
        go.Scatter(
            x=[shot.x for shot in home_shots],
            y=[shot.y for shot in home_shots],
            mode="markers",
            marker=dict(
                color=home_color,
                size=[10 + (math.sqrt(shot.xg) * 10 if shot.xg > 0 else 8) for shot in home_shots],
                symbol=["star" if shot.goal else "circle" for shot in home_shots],
                line=dict(color="white", width=1.5),
                opacity=0.85,
            ),
            name="Home shots",
            text=[
                f"{shot.minute:02d}:{shot.second:02d} — xG {shot.xg:.2f} ({shot.outcome})"
                for shot in home_shots
            ],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[shot.x for shot in away_shots],
            y=[shot.y for shot in away_shots],
            mode="markers",
            marker=dict(
                color=away_color,
                size=[10 + (math.sqrt(shot.xg) * 10 if shot.xg > 0 else 8) for shot in away_shots],
                symbol=["star" if shot.goal else "circle" for shot in away_shots],
                line=dict(color="white", width=1.5),
                opacity=0.85,
            ),
            name="Away shots",
            text=[
                f"{shot.minute:02d}:{shot.second:02d} — xG {shot.xg:.2f} ({shot.outcome})"
                for shot in away_shots
            ],
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    first_event = events_with_frames[0]
    first_freeze = freeze_lookup[first_event.event_uuid]
    home_players, away_players = split_freeze_players(first_freeze, home_color, away_color)
    ball_position = infer_ball_position(first_event, first_freeze)

    fig.add_trace(home_players, row=1, col=1)
    fig.add_trace(away_players, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=[ball_position[0]] if ball_position[0] is not None else [None],
            y=[ball_position[1]] if ball_position[1] is not None else [None],
            mode="markers",
            marker=dict(color="#fefefe", size=16, line=dict(color="black", width=2)),
            name="Ball",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[60],
            y=[-5],
            mode="text",
            text=[build_event_summary(first_event, teams)],
            textfont=dict(color="white", size=14),
            showlegend=False,
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="text",
            text=[build_scoreboard_text(first_event, teams)],
            textfont=dict(color="white", size=14),
            showlegend=False,
            hoverinfo="none",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[first_event.match_time_minutes],
            y=[0],
            mode="markers",
            marker=dict(color="#ffff00", size=20, symbol="line-ns-open"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(range=[-1, 1], visible=False, row=1, col=2)
    fig.update_yaxes(range=[-1, 1], visible=False, row=1, col=2)

    frames = []
    slider_steps = []
    for idx, event in enumerate(events_with_frames):
        freeze_records_for_event = freeze_lookup[event.event_uuid]
        home_trace, away_trace = split_freeze_players(
            freeze_records_for_event, home_color, away_color
        )
        ball_x, ball_y = infer_ball_position(event, freeze_records_for_event)
        frame = go.Frame(
            data=[
                home_trace,
                away_trace,
                go.Scatter(
                    x=[ball_x] if ball_x is not None else [None],
                    y=[ball_y] if ball_y is not None else [None],
                    mode="markers",
                    marker=dict(color="#fefefe", size=16, line=dict(color="black", width=2)),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=[60],
                    y=[-5],
                    mode="text",
                    text=[build_event_summary(event, teams)],
                    textfont=dict(color="white", size=14),
                    showlegend=False,
                    hoverinfo="none",
                ),
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="text",
                    text=[build_scoreboard_text(event, teams)],
                    textfont=dict(color="white", size=14),
                    showlegend=False,
                    hoverinfo="none",
                ),
                go.Scatter(
                    x=[event.match_time_minutes],
                    y=[0],
                    mode="markers",
                    marker=dict(color="#ffff00", size=20, symbol="line-ns-open"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ],
            name=str(idx),
            traces=[3, 4, 5, 6, 7, 8],
        )
        frames.append(frame)
        slider_steps.append(
            {
                "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": f"{event.minute:02d}:{event.second:02d} {event.event_type}",
                "method": "animate",
            }
        )

    fig.frames = frames

    fig.update_layout(
        width=1200,
        height=800,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        plot_bgcolor="#0b6623",
        paper_bgcolor="#123524",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=80, b=40),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Event: ", font=dict(color="white", size=16)),
                pad=dict(t=30),
                steps=slider_steps,
                x=0.1,
                len=0.7,
            )
        ],
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[[frame.name for frame in frames], {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
                pad=dict(t=30, r=10),
                x=0.85,
                y=1.07,
                bgcolor="#1b4332",
            )
        ],
        annotations=[
            dict(
                text=(
                    f"Referee: {match_info.get('referee', {}).get('name', 'Unknown')}<br>"
                    f"Venue: {match_info.get('stadium', {}).get('name', 'Unknown')}"
                ),
                x=0.82,
                y=0.32,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="white"),
            )
        ],
    )

    return fig


def build_three_sixty_catalog(manager: DataManager) -> Dict[int, dict]:
    match_ids = set(manager.list_three_sixty_match_ids())
    competitions = manager.get_competitions()
    catalog: Dict[int, dict] = {}
    remaining = set(match_ids)
    for competition in competitions:
        competition_id = competition.get("competition_id")
        season_id = competition.get("season_id")
        if competition_id is None or season_id is None:
            continue
        matches = manager.get_matches(int(competition_id), int(season_id))
        for match in matches:
            match_id = match.get("match_id")
            if match_id in remaining:
                catalog[int(match_id)] = {
                    "match": match,
                    "competition": competition,
                    "competition_id": int(competition_id),
                    "season_id": int(season_id),
                }
                remaining.remove(match_id)
        if not remaining:
            break
    return catalog


def list_matches_with_three_sixty(
    catalog: Dict[int, dict],
    competition_id: Optional[int] = None,
    season_id: Optional[int] = None,
    team_query: Optional[str] = None,
) -> None:
    team_query_normalised = team_query.lower() if team_query else None
    rows = []
    for match_id, entry in catalog.items():
        comp_id = entry["competition_id"]
        comp_season = entry["season_id"]
        if competition_id is not None and comp_id != competition_id:
            continue
        if season_id is not None and comp_season != season_id:
            continue
        match = entry["match"]
        home = match.get("home_team", {}).get("home_team_name", "Unknown")
        away = match.get("away_team", {}).get("away_team_name", "Unknown")
        if team_query_normalised and team_query_normalised not in home.lower() and team_query_normalised not in away.lower():
            continue
        rows.append(
            (
                match_id,
                match.get("match_date"),
                home,
                away,
                match.get("competition_stage", {}).get("name"),
                entry["competition"].get("competition_name"),
                entry["competition"].get("season_name"),
            )
        )
    if not rows:
        print("No matches with 360 data found that match the given filters.")
        return
    rows.sort(key=lambda item: (item[1] or "", item[0]))
    header = (
        f"{'Match ID':>8}  {'Date':<12}  {'Home':<25}  {'Away':<25}  {'Stage':<15}  "
        f"{'Competition':<25}  {'Season'}"
    )
    print(header)
    print("-" * len(header))
    for match_id, date, home, away, stage, competition_name, season_name in rows:
        print(
            f"{match_id:>8}  {str(date):<12}  {home:<25}  {away:<25}  {str(stage):<15}  "
            f"{str(competition_name):<25}  {str(season_name)}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive replay for a StatsBomb 360 match.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        help="Optional StatsBomb competition id filter when listing matches",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        help="Optional StatsBomb season id filter when listing matches",
    )
    parser.add_argument("--match-id", type=int, help="StatsBomb match id")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("match_replay.html"),
        help="Path to the HTML file that will contain the replay",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory where downloaded and processed data will be stored",
    )
    parser.add_argument(
        "--list-matches",
        action="store_true",
        help="List all matches with 360 data (optionally filtered) and exit",
    )
    parser.add_argument(
        "--team",
        type=str,
        help="Filter listed matches by team name (case insensitive substring)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    manager = DataManager(args.data_dir)
    catalog = build_three_sixty_catalog(manager)

    if args.list_matches:
        if args.match_id:
            print("--match-id is ignored when --list-matches is supplied.")
        list_matches_with_three_sixty(
            catalog,
            competition_id=args.competition_id,
            season_id=args.season_id,
            team_query=args.team,
        )
        return

    if args.match_id is None:
        raise SystemExit("--match-id is required unless --list-matches is used")

    match_entry = catalog.get(args.match_id)
    if match_entry is None:
        raise SystemExit(
            "Match not found among StatsBomb 360 fixtures. Run with --list-matches to discover available ids."
        )

    match_info = match_entry["match"]

    freeze_data = manager.get_three_sixty(args.match_id)
    if not freeze_data:
        raise SystemExit("No 360 data available for this match")

    events_data = manager.get_events(args.match_id)
    lineups = manager.get_lineups(args.match_id)

    teams = build_team_directory(match_info)
    players = build_player_directory(lineups, teams)
    freeze_lookup = freeze_frames_by_event(freeze_data)
    events, freeze_frames, shots = process_events(events_data, freeze_lookup, players, teams)

    manager.save_processed_dataset(args.match_id, events, freeze_frames)

    figure = build_replay_figure(events, freeze_frames, shots, teams, match_info)
    figure.write_html(str(args.output), include_plotlyjs="cdn")
    print(f"Replay saved to {args.output}")


if __name__ == "__main__":
    main()
