"""Interactive StatsBomb 360 match replay generator.

This script downloads StatsBomb open-data assets, caches them locally, and
turns freeze-frame events into an interactive Plotly animation that can be
saved as a standalone HTML file. The output keeps the pitch orientation
consistent, renders every player as a coloured dot, highlights the ball
carrier, and surfaces contextual details such as possession changes, goals,
and shot quality. Alongside the visualisation, the processed freeze-frame
stream is exported to JSON so it can seed future analytical workflows such as
Markov decision process modelling.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests import Response

STATS_BOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE_DIR = Path("statsbomb_cache")

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

HOME_COLOUR = "#1f77b4"
AWAY_COLOUR = "#ff4136"
BALL_COLOUR = "#f1c40f"
TIMELINE_NEUTRAL = "#bbbbbb"

GOAL_OUTCOMES = {"Goal"}
ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved To Post"}


@dataclass
class MatchSummary:
    """Metadata describing a match that exposes StatsBomb 360 data."""

    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    match_id: int
    match_date: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str
    freeze_frame_events: int



def fetch_statsbomb(path: str, allow_missing: bool = False) -> Optional[Iterable]:
    """Download and cache a JSON document from the StatsBomb open data set."""

    cache_path = CACHE_DIR / path
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    url = f"{STATS_BOMB_BASE}/{path}"
    response: Response = requests.get(url, timeout=30)
    if response.status_code == 404:
        if allow_missing:
            return None
        response.raise_for_status()
    response.raise_for_status()

    data = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return data


def load_competitions() -> List[dict]:
    competitions = fetch_statsbomb("competitions.json")
    if competitions is None:
        raise RuntimeError("Unable to download competitions list from StatsBomb.")
    return list(competitions)


def load_matches(competition_id: int, season_id: int) -> List[dict]:
    matches = fetch_statsbomb(f"matches/{competition_id}/{season_id}.json")
    return list(matches or [])


def load_events(match_id: int) -> List[dict]:
    events = fetch_statsbomb(f"events/{match_id}.json")
    return list(events or [])


def load_freeze_frames(match_id: int) -> Dict[str, List[dict]]:
    frames = fetch_statsbomb(f"three-sixty/{match_id}.json", allow_missing=True)
    if not frames:
        return {}

    mapping: Dict[str, List[dict]] = {}
    for entry in frames:
        event_uuid = entry.get("event_uuid")
        if not event_uuid:
            continue
        mapping[event_uuid] = entry.get("freeze_frame", [])
    return mapping


def load_lineups(match_id: int) -> Dict[int, int]:
    """Return a mapping of player_id -> team_id for the selected match."""

    lineups = fetch_statsbomb(f"lineups/{match_id}.json")
    if not lineups:
        raise RuntimeError(
            "Lineups data is required to map players to teams but was not available."
        )

    mapping: Dict[int, int] = {}
    for team in lineups:
        team_id = int(team.get("team_id"))
        for player in team.get("lineup", []):
            player_id = player.get("player_id")
            if player_id is None:
                continue
            mapping[int(player_id)] = team_id
    return mapping


def discover_three_sixty_matches(
    *,
    competition_filter: Optional[int] = None,
    season_filter: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[MatchSummary]:
    """Return matches that expose StatsBomb 360 freeze frames."""

    summaries: List[MatchSummary] = []
    for competition in load_competitions():
        comp_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        if competition_filter is not None and comp_id != competition_filter:
            continue
        if season_filter is not None and season_id != season_filter:
            continue

        for match in load_matches(comp_id, season_id):
            match_id = int(match["match_id"])
            freeze_frames = fetch_statsbomb(
                f"three-sixty/{match_id}.json", allow_missing=True
            )
            if not freeze_frames:
                continue

            freeze_event_count = sum(
                1 for frame in freeze_frames if frame.get("freeze_frame")
            )
            summaries.append(
                MatchSummary(
                    competition_id=comp_id,
                    competition_name=competition.get("competition_name", ""),
                    season_id=season_id,
                    season_name=competition.get("season_name", ""),
                    match_id=match_id,
                    match_date=match.get("match_date", ""),
                    home_team_id=int(match.get("home_team", {}).get("home_team_id", 0)),
                    home_team_name=match.get("home_team", {}).get("home_team_name", ""),
                    away_team_id=int(match.get("away_team", {}).get("away_team_id", 0)),
                    away_team_name=match.get("away_team", {}).get("away_team_name", ""),
                    freeze_frame_events=freeze_event_count,
                )
            )
            if limit and len(summaries) >= limit:
                return summaries
    return summaries


def format_clock(minute: int, second: int) -> str:
    return f"{int(minute):02d}:{int(second):02d}"


def describe_event(event: dict) -> str:
    minute = int(event.get("minute", 0))
    second = int(event.get("second", 0))
    player_name = event.get("player", {}).get("name") or "Unknown player"
    event_type = event.get("type", {}).get("name") or "Action"
    team_name = event.get("possession_team", {}).get("name") or "Unknown team"

    details = ""
    if event_type == "Shot":
        shot = event.get("shot", {})
        outcome = shot.get("outcome", {}).get("name")
        xg = shot.get("statsbomb_xg")
        if outcome:
            details = f" ({outcome})"
        if xg:
            details += f" xG={float(xg):.2f}"
    elif event_type == "Pass":
        passed = event.get("pass", {})
        if passed.get("goal_assist"):
            details = " (assist)"
        elif passed.get("shot_assist"):
            details = " (shot assist)"
    elif event_type == "Foul Won":
        details = " (foul won)"

    return f"{format_clock(minute, second)} — {team_name}: {player_name} {event_type}{details}"


def apply_orientation(value: Sequence[float], flip: bool) -> Tuple[Optional[float], Optional[float]]:
    if not value or len(value) < 2:
        return None, None
    x, y = value[0], value[1]
    if x is None or y is None:
        return None, None
    x = float(x)
    y = float(y)
    if flip:
        x = PITCH_LENGTH - x
    return x, y


def determine_orientation(
    freeze_frames: Dict[str, List[dict]],
    player_to_team: Dict[int, int],
    home_team_id: int,
) -> bool:
    """Determine if coordinates need flipping to keep the home team on the left."""

    for frame in freeze_frames.values():
        home_x: List[float] = []
        away_x: List[float] = []
        for player in frame:
            player_info = player.get("player", {})
            player_id = player_info.get("id") or player_info.get("player_id")
            location = player.get("location")
            if player_id is None or not location:
                continue
            team_id = player_to_team.get(int(player_id))
            if team_id is None:
                continue
            x_value = float(location[0])
            if team_id == home_team_id:
                home_x.append(x_value)
            else:
                away_x.append(x_value)
        if home_x and away_x:
            home_avg = sum(home_x) / len(home_x)
            away_avg = sum(away_x) / len(away_x)
            return home_avg > away_avg
    return False


def build_timeline_data(
    events: List[dict],
    freeze_frames: Dict[str, List[dict]],
) -> List[dict]:
    return [event for event in events if event.get("id") in freeze_frames]


def compute_score_update(
    event: dict,
    home_team_id: int,
    score_home: int,
    score_away: int,
) -> Tuple[int, int]:
    if event.get("type", {}).get("name") != "Shot":
        return score_home, score_away
    shot = event.get("shot", {})
    outcome = shot.get("outcome", {}).get("name")
    if outcome not in GOAL_OUTCOMES:
        return score_home, score_away
    team_id = event.get("team", {}).get("id")
    if team_id == home_team_id:
        return score_home + 1, score_away
    return score_home, score_away + 1


def update_shot_metrics(
    event: dict,
    shots: Dict[int, int],
    shots_on_target: Dict[int, int],
    xg_totals: Dict[int, float],
) -> None:
    if event.get("type", {}).get("name") != "Shot":
        return
    team_id = event.get("team", {}).get("id")
    if team_id is None:
        return
    shots[team_id] = shots.get(team_id, 0) + 1
    shot = event.get("shot", {})
    outcome = shot.get("outcome", {}).get("name")
    if outcome in ON_TARGET_OUTCOMES:
        shots_on_target[team_id] = shots_on_target.get(team_id, 0) + 1
    xg = shot.get("statsbomb_xg")
    if xg is not None:
        xg_totals[team_id] = xg_totals.get(team_id, 0.0) + float(xg)


def extract_positions(
    freeze_frame: List[dict],
    player_to_team: Dict[int, int],
    home_team_id: int,
    flip: bool,
    ball_carrier_id: Optional[int],
) -> Tuple[List[float], List[float], List[str], List[float], List[float], List[float], List[str], List[float]]:
    """Return home/away player coordinates and marker sizes."""

    home_x: List[float] = []
    home_y: List[float] = []
    home_labels: List[str] = []
    home_sizes: List[float] = []

    away_x: List[float] = []
    away_y: List[float] = []
    away_labels: List[str] = []
    away_sizes: List[float] = []

    for player in freeze_frame:
        player_info = player.get("player", {})
        player_id = player_info.get("id") or player_info.get("player_id")
        if player_id is None:
            continue
        location = player.get("location")
        x, y = apply_orientation(location or (), flip)
        if x is None or y is None:
            continue
        team_id = player_to_team.get(int(player_id))
        if team_id is None:
            continue
        label = player_info.get("name") or "Unknown"
        size = 16.0 if ball_carrier_id is not None and int(player_id) == ball_carrier_id else 11.0
        if team_id == home_team_id:
            home_x.append(x)
            home_y.append(y)
            home_labels.append(label)
            home_sizes.append(size)
        else:
            away_x.append(x)
            away_y.append(y)
            away_labels.append(label)
            away_sizes.append(size)
    return home_x, home_y, home_labels, home_sizes, away_x, away_y, away_labels, away_sizes


def prepare_animation_frames(
    ordered_events: List[dict],
    freeze_frames: Dict[str, List[dict]],
    player_to_team: Dict[int, int],
    home_team_id: int,
    away_team_id: int,
    flip: bool,
    home_team_name: str,
    away_team_name: str,
) -> Tuple[List[dict], List[dict]]:
    """Return frame metadata and timeline markers for the animation."""

    frames: List[dict] = []
    timeline_markers: List[dict] = []

    score_home = 0
    score_away = 0
    shots: Dict[int, int] = {home_team_id: 0, away_team_id: 0}
    shots_on_target: Dict[int, int] = {home_team_id: 0, away_team_id: 0}
    xg_totals: Dict[int, float] = {home_team_id: 0.0, away_team_id: 0.0}

    filtered_events = build_timeline_data(ordered_events, freeze_frames)

    for idx, event in enumerate(filtered_events):
        event_id = event.get("id")
        if not event_id:
            continue
        freeze_frame = freeze_frames.get(event_id)
        if not freeze_frame:
            continue

        player_info = event.get("player", {})
        ball_carrier_id = player_info.get("id") or player_info.get("player_id")

        (
            home_x,
            home_y,
            home_labels,
            home_sizes,
            away_x,
            away_y,
            away_labels,
            away_sizes,
        ) = extract_positions(
            freeze_frame,
            player_to_team,
            home_team_id,
            flip,
            int(ball_carrier_id) if ball_carrier_id is not None else None,
        )

        ball_x, ball_y = apply_orientation(event.get("location") or (), flip)

        score_home, score_away = compute_score_update(event, home_team_id, score_home, score_away)
        update_shot_metrics(event, shots, shots_on_target, xg_totals)

        minute = int(event.get("minute", 0))
        second = int(event.get("second", 0))
        time_seconds = minute * 60 + second
        possession_team_id = event.get("possession_team", {}).get("id")

        next_possession_id: Optional[int] = None
        if idx + 1 < len(filtered_events):
            next_possession_id = filtered_events[idx + 1].get("possession_team", {}).get("id")
        possession_changed = (
            possession_team_id is not None
            and next_possession_id is not None
            and possession_team_id != next_possession_id
        )

        event_type = event.get("type", {}).get("name", "Event")
        description = describe_event(event)

        shot = event.get("shot")
        shot_line: Tuple[List[float], List[float]] = ([], [])
        if shot:
            end_location = shot.get("end_location") or []
            end_x, end_y = apply_orientation(end_location, flip)
            if ball_x is not None and ball_y is not None and end_x is not None and end_y is not None:
                shot_line = ([ball_x, end_x], [ball_y, end_y])

        score_text = (
            f"<b>{home_team_name}</b> {score_home} - {score_away} <b>{away_team_name}</b>"
        )
        stats_text = (
            f"Shots: {shots.get(home_team_id, 0)} - {shots.get(away_team_id, 0)}"\
            f"<br>On target: {shots_on_target.get(home_team_id, 0)} - {shots_on_target.get(away_team_id, 0)}"\
            f"<br>xG: {xg_totals.get(home_team_id, 0.0):.2f} - {xg_totals.get(away_team_id, 0.0):.2f}"
        )

        carrier_name = player_info.get("name") or "Unknown"
        possession_note = " — possession lost" if possession_changed else ""
        event_label = f"{format_clock(minute, second)} {carrier_name}: {event_type}{possession_note}"

        frames.append(
            {
                "home_x": home_x,
                "home_y": home_y,
                "home_labels": home_labels,
                "home_sizes": home_sizes,
                "away_x": away_x,
                "away_y": away_y,
                "away_labels": away_labels,
                "away_sizes": away_sizes,
                "ball_x": [ball_x] if ball_x is not None else [],
                "ball_y": [ball_y] if ball_y is not None else [],
                "description": description,
                "event_label": event_label,
                "score_text": score_text,
                "stats_text": stats_text,
                "shot_line": shot_line,
                "time_seconds": time_seconds,
                "team_id": event.get("team", {}).get("id"),
                "event_type": event_type,
                "is_goal": event.get("shot", {}).get("outcome", {}).get("name") in GOAL_OUTCOMES,
                "is_shot": event_type == "Shot",
            }
        )

        timeline_markers.append(
            {
                "time": time_seconds,
                "team_id": event.get("team", {}).get("id"),
                "label": event_type,
                "description": description,
                "is_goal": frames[-1]["is_goal"],
                "is_shot": frames[-1]["is_shot"],
            }
        )

    return frames, timeline_markers


def save_processed_stream(
    match: MatchSummary,
    frames: List[dict],
    timeline_markers: List[dict],
    output_dir: Path,
) -> Path:
    """Persist the processed freeze-frame stream for future analysis."""

    payload = {
        "match": {
            "match_id": match.match_id,
            "match_date": match.match_date,
            "competition_id": match.competition_id,
            "competition_name": match.competition_name,
            "season_id": match.season_id,
            "season_name": match.season_name,
            "home_team_id": match.home_team_id,
            "home_team_name": match.home_team_name,
            "away_team_id": match.away_team_id,
            "away_team_name": match.away_team_name,
        },
        "frames": frames,
        "timeline": timeline_markers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"match_{match.match_id}_freeze_frames.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output_path


def pitch_shapes() -> List[dict]:
    """Return Plotly shape definitions for a stylised football pitch."""

    shapes = [
        {
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": PITCH_LENGTH,
            "y1": PITCH_WIDTH,
            "line": {"color": "#ffffff", "width": 2},
        },
        {
            "type": "line",
            "x0": PITCH_LENGTH / 2,
            "y0": 0,
            "x1": PITCH_LENGTH / 2,
            "y1": PITCH_WIDTH,
            "line": {"color": "#ffffff", "width": 1},
        },
        {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": PITCH_LENGTH / 2 - 9.15,
            "y0": PITCH_WIDTH / 2 - 9.15,
            "x1": PITCH_LENGTH / 2 + 9.15,
            "y1": PITCH_WIDTH / 2 + 9.15,
            "line": {"color": "#ffffff", "width": 1},
        },
    ]

    penalty_length = 16.5
    penalty_width = 40.32
    six_length = 5.5
    six_width = 18.32

    shapes.extend(
        [
            {
                "type": "rect",
                "x0": 0,
                "y0": (PITCH_WIDTH - penalty_width) / 2,
                "x1": penalty_length,
                "y1": (PITCH_WIDTH + penalty_width) / 2,
                "line": {"color": "#ffffff", "width": 1},
            },
            {
                "type": "rect",
                "x0": PITCH_LENGTH - penalty_length,
                "y0": (PITCH_WIDTH - penalty_width) / 2,
                "x1": PITCH_LENGTH,
                "y1": (PITCH_WIDTH + penalty_width) / 2,
                "line": {"color": "#ffffff", "width": 1},
            },
            {
                "type": "rect",
                "x0": 0,
                "y0": (PITCH_WIDTH - six_width) / 2,
                "x1": six_length,
                "y1": (PITCH_WIDTH + six_width) / 2,
                "line": {"color": "#ffffff", "width": 1},
            },
            {
                "type": "rect",
                "x0": PITCH_LENGTH - six_length,
                "y0": (PITCH_WIDTH - six_width) / 2,
                "x1": PITCH_LENGTH,
                "y1": (PITCH_WIDTH + six_width) / 2,
                "line": {"color": "#ffffff", "width": 1},
            },
        ]
    )
    return shapes


def build_timeline_trace(timeline_markers: List[dict], home_team_id: int, away_team_id: int) -> go.Scatter:
    x_values = [marker["time"] for marker in timeline_markers]
    y_values = []
    marker_colours = []
    marker_symbols = []
    marker_sizes = []
    hover_texts = []

    for marker in timeline_markers:
        team_id = marker.get("team_id")
        if team_id == home_team_id:
            y_values.append(0)
            marker_colours.append(HOME_COLOUR)
        elif team_id == away_team_id:
            y_values.append(1)
            marker_colours.append(AWAY_COLOUR)
        else:
            y_values.append(0.5)
            marker_colours.append(TIMELINE_NEUTRAL)

        if marker.get("is_goal"):
            marker_symbols.append("star")
            marker_sizes.append(14)
        elif marker.get("is_shot"):
            marker_symbols.append("triangle-up")
            marker_sizes.append(11)
        else:
            marker_symbols.append("circle")
            marker_sizes.append(9)
        hover_texts.append(marker.get("description", ""))

    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers",
        marker=dict(color=marker_colours, size=marker_sizes, symbol=marker_symbols),
        hovertemplate="%{text}<extra></extra>",
        text=hover_texts,
        name="Event timeline",
    )


def make_animation(
    match: MatchSummary,
    frames: List[dict],
    timeline_markers: List[dict],
    output_path: Path,
    playback_duration: int,
) -> None:
    if not frames:
        raise RuntimeError("No freeze-frame events were found for this match.")

    timeline_trace = build_timeline_trace(
        timeline_markers, match.home_team_id, match.away_team_id
    )

    timeline_x = list(timeline_trace.x)
    timeline_y = list(timeline_trace.y)
    timeline_colors = list(timeline_trace.marker.color)
    timeline_symbols = list(timeline_trace.marker.symbol)
    base_marker_sizes = [
        14 if marker.get("is_goal") else 11 if marker.get("is_shot") else 9
        for marker in timeline_markers
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
    )

    initial = frames[0]

    home_trace = go.Scatter(
        x=initial["home_x"],
        y=initial["home_y"],
        mode="markers",
        marker=dict(color=HOME_COLOUR, size=initial["home_sizes"], line=dict(width=1, color="#ffffff")),
        text=initial["home_labels"],
        hovertemplate="%{text}<extra></extra>",
        name=match.home_team_name,
    )

    away_trace = go.Scatter(
        x=initial["away_x"],
        y=initial["away_y"],
        mode="markers",
        marker=dict(color=AWAY_COLOUR, size=initial["away_sizes"], line=dict(width=1, color="#ffffff")),
        text=initial["away_labels"],
        hovertemplate="%{text}<extra></extra>",
        name=match.away_team_name,
    )

    ball_trace = go.Scatter(
        x=initial["ball_x"],
        y=initial["ball_y"],
        mode="markers",
        marker=dict(color=BALL_COLOUR, size=18, symbol="circle"),
        hoverinfo="skip",
        name="Ball",
    )

    shot_trace = go.Scatter(
        x=initial["shot_line"][0],
        y=initial["shot_line"][1],
        mode="lines",
        line=dict(color="#ffffff", width=2, dash="dot"),
        hoverinfo="skip",
        name="Shot path",
    )

    fig.add_trace(home_trace, row=1, col=1)
    fig.add_trace(away_trace, row=1, col=1)
    fig.add_trace(ball_trace, row=1, col=1)
    fig.add_trace(shot_trace, row=1, col=1)
    fig.add_trace(timeline_trace, row=2, col=1)

    initial_annotation_text = (
        f"<b>{initial['event_label']}</b><br>{initial['description']}"
    )
    initial_annotations = [
        dict(
            x=1.03,
            y=0.82,
            xref="paper",
            yref="paper",
            text=initial["score_text"],
            showarrow=False,
            align="left",
            font=dict(color="#ffffff", size=18),
        ),
        dict(
            x=1.03,
            y=0.48,
            xref="paper",
            yref="paper",
            text=initial["stats_text"],
            showarrow=False,
            align="left",
            font=dict(color="#dddddd", size=12),
        ),
        dict(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            text=initial_annotation_text,
            showarrow=False,
            align="left",
            font=dict(color="#ffffff", size=13),
        ),
    ]

    slider_steps = []
    animation_frames = []
    for idx, frame in enumerate(frames):
        step = dict(
            method="animate",
            label=str(idx + 1),
            args=[[str(idx)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
        )
        slider_steps.append(step)

        annotation_text = (
            f"<b>{frame['event_label']}</b><br>{frame['description']}"
        )

        frame_annotations = [
            dict(
                x=1.03,
                y=0.82,
                xref="paper",
                yref="paper",
                text=frame["score_text"],
                showarrow=False,
                align="left",
                font=dict(color="#ffffff", size=18),
            ),
            dict(
                x=1.03,
                y=0.48,
                xref="paper",
                yref="paper",
                text=frame["stats_text"],
                showarrow=False,
                align="left",
                font=dict(color="#dddddd", size=12),
            ),
            dict(
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                text=annotation_text,
                showarrow=False,
                align="left",
                font=dict(color="#ffffff", size=13),
            ),
        ]

        highlight_sizes = []
        for marker_idx, base_size in enumerate(base_marker_sizes):
            if marker_idx == idx:
                highlight_sizes.append(base_size + 4)
            else:
                highlight_sizes.append(base_size)

        animation_frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(
                        x=frame["home_x"],
                        y=frame["home_y"],
                        mode="markers",
                        marker=dict(
                            size=frame["home_sizes"],
                            color=HOME_COLOUR,
                            line=dict(width=1, color="#ffffff"),
                        ),
                        text=frame["home_labels"],
                    ),
                    go.Scatter(
                        x=frame["away_x"],
                        y=frame["away_y"],
                        mode="markers",
                        marker=dict(
                            size=frame["away_sizes"],
                            color=AWAY_COLOUR,
                            line=dict(width=1, color="#ffffff"),
                        ),
                        text=frame["away_labels"],
                    ),
                    go.Scatter(
                        x=frame["ball_x"],
                        y=frame["ball_y"],
                        mode="markers",
                        marker=dict(color=BALL_COLOUR, size=18),
                    ),
                    go.Scatter(
                        x=frame["shot_line"][0],
                        y=frame["shot_line"][1],
                        mode="lines",
                    ),
                    go.Scatter(
                        x=timeline_x,
                        y=timeline_y,
                        mode="markers",
                        marker=dict(
                            color=timeline_colors,
                            symbol=timeline_symbols,
                            size=highlight_sizes,
                        ),
                        text=timeline_trace.text,
                        hovertemplate=timeline_trace.hovertemplate,
                        name=timeline_trace.name,
                    ),
                ],
                layout=go.Layout(annotations=frame_annotations),
            )
        )

    fig.frames = animation_frames

    fig.update_layout(
        title=(
            f"{match.home_team_name} vs {match.away_team_name} ({match.match_date})<br>"
            f"{match.competition_name} {match.season_name}"
        ),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        width=1200,
        height=820,
        showlegend=False,
        margin=dict(l=50, r=320, t=90, b=40),
        shapes=pitch_shapes(),
        xaxis=dict(
            range=[0, PITCH_LENGTH],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[0, PITCH_WIDTH],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        xaxis2=dict(title="Match seconds", showgrid=True, zeroline=False, color="#bbbbbb"),
        yaxis2=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=[match.home_team_name, match.away_team_name],
            range=[-0.5, 1.5],
            showgrid=False,
            zeroline=False,
            color="#bbbbbb",
        ),
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Frame ", "font": {"color": "#ffffff"}},
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": playback_duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": max(playback_duration // 3, 60)},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 50},
                "x": 0.1,
                "y": -0.07,
                "bgcolor": "#222222",
                "bordercolor": "#444444",
                "font": {"color": "#ffffff"},
            }
        ],
        annotations=initial_annotations,
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_play=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly replay for StatsBomb 360 matches.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        help="Limit search to a specific competition ID.",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        help="Limit search to a specific season ID.",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        help="Render a specific match (must expose 360 data).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=5,
        help="Maximum matches to scan when listing options.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("match_replay.html"),
        help="Destination HTML file for the replay animation.",
    )
    parser.add_argument(
        "--playback-speed",
        type=int,
        default=500,
        help="Frame duration in milliseconds when playing the animation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matches that expose 360 freeze frames and exit.",
    )
    return parser.parse_args()


def select_match(matches: List[MatchSummary], match_id: Optional[int]) -> MatchSummary:
    if match_id is None:
        if not matches:
            raise RuntimeError(
                "No matches with StatsBomb 360 data were found. Increase --max-matches or adjust filters."
            )
        return matches[0]

    for summary in matches:
        if summary.match_id == match_id:
            return summary

    freeze_frames = load_freeze_frames(match_id)
    if not freeze_frames:
        raise RuntimeError(
            f"Match {match_id} does not expose StatsBomb 360 data. Choose another fixture."
        )

    for competition in load_competitions():
        comp_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        for match in load_matches(comp_id, season_id):
            if int(match["match_id"]) == match_id:
                return MatchSummary(
                    competition_id=comp_id,
                    competition_name=competition.get("competition_name", ""),
                    season_id=season_id,
                    season_name=competition.get("season_name", ""),
                    match_id=match_id,
                    match_date=match.get("match_date", ""),
                    home_team_id=int(match.get("home_team", {}).get("home_team_id", 0)),
                    home_team_name=match.get("home_team", {}).get("home_team_name", ""),
                    away_team_id=int(match.get("away_team", {}).get("away_team_id", 0)),
                    away_team_name=match.get("away_team", {}).get("away_team_name", ""),
                    freeze_frame_events=sum(
                        1 for frame in freeze_frames.values() if frame
                    ),
                )
    raise RuntimeError(
        "Unable to locate metadata for the requested match even though 360 data exists."
    )


def main() -> None:
    args = parse_args()

    matches = discover_three_sixty_matches(
        competition_filter=args.competition_id,
        season_filter=args.season_id,
        limit=args.max_matches,
    )

    if args.list:
        if not matches:
            print("No matches with StatsBomb 360 data were found for the given filters.")
            return
        print("Matches with StatsBomb 360 data:")
        for summary in matches:
            print(
                f"match_id={summary.match_id} | {summary.match_date} | "
                f"{summary.home_team_name} vs {summary.away_team_name} | "
                f"{summary.competition_name} {summary.season_name} | "
                f"freeze frames: {summary.freeze_frame_events}"
            )
        return

    match = select_match(matches, args.match_id)

    freeze_frames = load_freeze_frames(match.match_id)
    if not freeze_frames:
        raise RuntimeError(
            f"Match {match.match_id} no longer exposes StatsBomb 360 data."
        )

    events = load_events(match.match_id)
    player_map = load_lineups(match.match_id)

    flip = determine_orientation(freeze_frames, player_map, match.home_team_id)

    frames, timeline_markers = prepare_animation_frames(
        events,
        freeze_frames,
        player_map,
        match.home_team_id,
        match.away_team_id,
        flip,
        match.home_team_name,
        match.away_team_name,
    )

    processed_path = save_processed_stream(
        match,
        frames,
        timeline_markers,
        CACHE_DIR / "processed",
    )

    make_animation(match, frames, timeline_markers, args.output, args.playback_speed)

    print(f"Replay saved to {args.output.resolve()}")
    print(f"Processed freeze-frame data cached at {processed_path.resolve()}")


if __name__ == "__main__":
    main()

