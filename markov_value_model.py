"""Interactive Markov chain visualiser built purely from StatsBomb 360 data.

The script downloads StatsBomb 360 freeze frames for matches that include them
and uses the player locations in each freeze frame to build a very transparent
Markov decision process (MDP) style value function:

* Each freeze-framed event provides a *state* represented by a grid covering the
  pitch. Grid cells accumulate positive weight for supporting teammates and
  negative weight for defenders. The ball carrier adds extra positive weight so
  the update focuses on the area of play.
* The value function starts as zeros and is updated sequentially with temporal
  difference learning so you can watch the grid evolve as events unfold.
* Rewards are derived from the event outcomes (shots, assists, possession
  changes) to keep the example intuitive.
* An interactive Plotly animation lets you step through the match, showing the
  updated value grid alongside the player and ball locations on a stylised pitch.

The workflow stays in a single file and requires only ``numpy``, ``plotly`` and
``requests`` so it is easy to reuse for presentations or workshops.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import requests
from requests import Response

STATS_BOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE_DIR = Path("statsbomb_cache")

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

ACTION_SPACE: Tuple[str, ...] = ("pass", "shoot", "dribble")


@dataclass
class MatchSummary:
    """Lightweight container for matches that ship with StatsBomb 360 data."""

    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    match_id: int
    home_team: str
    away_team: str
    match_date: str
    freeze_frame_events: int


def fetch_statsbomb(path: str, allow_missing: bool = False) -> Optional[Iterable]:
    """Download and cache a JSON document from the StatsBomb open-data repo."""

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
    """Return all competitions from the open-data set."""

    competitions = fetch_statsbomb("competitions.json")
    if competitions is None:
        raise RuntimeError("Unable to download competitions list from StatsBomb.")
    return list(competitions)


def load_matches(competition_id: int, season_id: int) -> List[dict]:
    """Return the matches for a competition/season pair."""

    matches = fetch_statsbomb(f"matches/{competition_id}/{season_id}.json")
    return list(matches or [])


def load_events(match_id: int) -> List[dict]:
    """Return the ordered StatsBomb events for a match."""

    events = fetch_statsbomb(f"events/{match_id}.json")
    return list(events or [])


def load_freeze_frames(match_id: int) -> Dict[str, List[dict]]:
    """Map event UUIDs to freeze-frame player locations for a match."""

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


def discover_three_sixty_matches(
    *,
    competition_filter: Optional[int] = None,
    season_filter: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[MatchSummary]:
    """Return matches that include StatsBomb 360 freeze frames."""

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
                    home_team=match.get("home_team", {}).get("home_team_name", ""),
                    away_team=match.get("away_team", {}).get("away_team_name", ""),
                    match_date=match.get("match_date", ""),
                    freeze_frame_events=freeze_event_count,
                )
            )
            if limit and len(summaries) >= limit:
                return summaries
    return summaries


def resolve_match_summary(
    match_id: int, known_matches: Sequence[MatchSummary]
) -> MatchSummary:
    """Return metadata for a match that includes StatsBomb 360 data."""

    for summary in known_matches:
        if summary.match_id == match_id:
            return summary

    freeze_frames = load_freeze_frames(match_id)
    if not freeze_frames:
        raise RuntimeError(
            f"Match {match_id} does not have StatsBomb 360 data available."
        )

    target_match: Optional[MatchSummary] = None
    for competition in load_competitions():
        comp_id = int(competition["competition_id"])
        season_id = int(competition["season_id"])
        matches_meta = load_matches(comp_id, season_id)
        for match in matches_meta:
            if int(match["match_id"]) == match_id:
                target_match = MatchSummary(
                    competition_id=comp_id,
                    competition_name=competition.get("competition_name", ""),
                    season_id=season_id,
                    season_name=competition.get("season_name", ""),
                    match_id=match_id,
                    home_team=match.get("home_team", {}).get("home_team_name", ""),
                    away_team=match.get("away_team", {}).get("away_team_name", ""),
                    match_date=match.get("match_date", ""),
                    freeze_frame_events=sum(1 for frame in freeze_frames.values() if frame),
                )
                break
        if target_match:
            break

    if target_match is None:
        raise RuntimeError(
            "Unable to locate metadata for the requested match, even though 360 data exists."
        )
    return target_match


def coordinate_to_cell(
    location: Sequence[float], grid_x: int, grid_y: int
) -> Optional[Tuple[int, int]]:
    """Convert a StatsBomb (x, y) location to a grid index (row, column)."""

    if not location or len(location) < 2:
        return None
    x, y = location[:2]
    if x is None or y is None:
        return None

    x = min(max(float(x), 0.0), PITCH_LENGTH - 1e-6)
    y = min(max(float(y), 0.0), PITCH_WIDTH - 1e-6)

    x_idx = int(x / (PITCH_LENGTH / grid_x))
    y_idx = int(y / (PITCH_WIDTH / grid_y))
    return y_idx, x_idx


def discretise_distance(distance: float, *, bin_size: float = 5.0, max_bin: int = 10) -> int:
    """Bucket a distance into an integer bin for Q-learning state aggregation."""

    if not np.isfinite(distance):
        return max_bin + 1
    bucket = int(distance // bin_size)
    return min(bucket, max_bin)


def count_players_in_grid(
    freeze_frame: List[dict], grid_x: int, grid_y: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return teammate/opponent occupancy counts for each grid cell."""

    teammates = np.zeros((grid_y, grid_x), dtype=int)
    opponents = np.zeros((grid_y, grid_x), dtype=int)
    for player in freeze_frame:
        cell = coordinate_to_cell(player.get("location"), grid_x, grid_y)
        if cell is None:
            continue
        row, col = cell
        if player.get("teammate"):
            teammates[row, col] += 1
        else:
            opponents[row, col] += 1
    return teammates, opponents


def nearest_opponent_distance(
    freeze_frame: List[dict], ball_location: Optional[Sequence[float]]
) -> float:
    """Return the Euclidean distance from the ball carrier to the closest opponent."""

    if not ball_location or len(ball_location) < 2:
        return float("inf")

    bx, by = float(ball_location[0]), float(ball_location[1])
    min_distance = float("inf")
    for player in freeze_frame:
        if player.get("teammate"):
            continue
        location = player.get("location")
        if not location or len(location) < 2:
            continue
        px, py = float(location[0]), float(location[1])
        distance = float(np.hypot(px - bx, py - by))
        if distance < min_distance:
            min_distance = distance
    return min_distance


def summarise_state(
    freeze_frame: List[dict],
    ball_location: Optional[Sequence[float]],
    grid_x: int,
    grid_y: int,
) -> Tuple[Tuple[int, ...], Dict[str, object]]:
    """Create a discrete state key and human-readable summary for Q-learning."""

    teammate_counts, opponent_counts = count_players_in_grid(freeze_frame, grid_x, grid_y)
    ball_cell = coordinate_to_cell(ball_location, grid_x, grid_y)
    nearest_distance = nearest_opponent_distance(freeze_frame, ball_location)
    distance_bin = discretise_distance(nearest_distance)

    if ball_cell is None:
        ball_cell = (-1, -1)

    state_key: Tuple[int, ...] = (
        ball_cell[0],
        ball_cell[1],
        distance_bin,
        *tuple(int(value) for value in teammate_counts.flatten()),
        *tuple(int(value) for value in opponent_counts.flatten()),
    )

    summary = {
        "ball_cell": ball_cell,
        "nearest_opponent_distance": None
        if not np.isfinite(nearest_distance)
        else float(nearest_distance),
        "teammate_counts": teammate_counts.tolist(),
        "opponent_counts": opponent_counts.tolist(),
        "distance_bin": distance_bin,
    }
    return state_key, summary


def build_context_grid(
    freeze_frame: List[dict],
    ball_location: Optional[Sequence[float]],
    grid_x: int,
    grid_y: int,
) -> np.ndarray:
    """Create a grid weighting teammates positively and defenders negatively."""

    grid = np.zeros((grid_y, grid_x), dtype=float)
    for player in freeze_frame:
        player_location = player.get("location")
        cell = coordinate_to_cell(player_location, grid_x, grid_y)
        if cell is None:
            continue
        row, col = cell
        weight = 1.0 if player.get("teammate") else -1.0
        # Goalkeepers are influential, give them extra magnitude.
        if player.get("keeper"):
            weight *= 1.5
        grid[row, col] += weight

    ball_cell = (
        coordinate_to_cell(ball_location, grid_x, grid_y)
        if ball_location is not None
        else None
    )
    if ball_cell is not None:
        row, col = ball_cell
        grid[row, col] += 2.0

    norm = float(np.sum(np.abs(grid)))
    if norm > 0.0:
        grid /= norm
    return grid


def describe_event(event: dict) -> str:
    """Return a human readable summary for an event."""

    minute = int(event.get("minute", 0))
    second = int(event.get("second", 0))
    player_name = event.get("player", {}).get("name") or "Unknown player"
    event_type = event.get("type", {}).get("name") or "Action"

    extra = ""
    if event_type == "Shot":
        shot = event.get("shot", {})
        outcome = shot.get("outcome", {}).get("name")
        if outcome:
            extra = f" ({outcome})"
    elif event_type == "Pass":
        passed = event.get("pass", {})
        if passed.get("goal_assist"):
            extra = " (assist)"
        elif passed.get("shot_assist"):
            extra = " (shot assist)"

    return f"{minute:02d}:{second:02d} - {player_name} {event_type}{extra}"


def infer_action(event: dict) -> Optional[str]:
    """Map a StatsBomb event to one of the discrete RL actions."""

    event_type = event.get("type", {}).get("name")
    if event_type == "Shot":
        return "shoot"
    if event_type == "Pass":
        return "pass"
    if event_type in {"Carry", "Dribble"}:
        return "dribble"
    return None


def extract_player_positions(freeze_frame: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Return teammate and opponent positions for plotting."""

    teammates: List[dict] = []
    opponents: List[dict] = []
    for player in freeze_frame:
        location = player.get("location")
        if not location or len(location) < 2:
            continue
        x, y = float(location[0]), float(location[1])
        point = {
            "x": x,
            "y": y,
            "name": player.get("player", {}).get("name", ""),
        }
        if player.get("teammate"):
            teammates.append(point)
        else:
            opponents.append(point)
    return teammates, opponents


def compute_reward(current_event: dict, next_event: Optional[dict]) -> float:
    """Simple reward signal based on event outcomes and possession changes."""

    reward = 0.0
    event_type = current_event.get("type", {}).get("name")

    if event_type == "Shot":
        shot = current_event.get("shot", {})
        outcome = shot.get("outcome", {}).get("name")
        if outcome == "Goal":
            reward = 1.0
        else:
            reward = float(shot.get("statsbomb_xg") or 0.0)
            if outcome in {"Saved", "Saved To Post"}:
                reward *= 0.5
            elif outcome in {"Off T", "Post"}:
                reward *= 0.2
    elif event_type == "Pass":
        passed = current_event.get("pass", {})
        if passed.get("goal_assist"):
            reward = 0.6
        elif passed.get("shot_assist"):
            reward = 0.3
        elif passed.get("switch"):
            reward = 0.1
    elif event_type in {"Ball Receipt", "Carry"}:
        reward = 0.05

    if next_event is not None:
        current_team = current_event.get("possession_team", {}).get("id")
        next_team = next_event.get("possession_team", {}).get("id")
        if current_team and next_team and current_team != next_team:
            reward -= 0.1
    return reward


def construct_timeline(
    events: List[dict],
    freeze_frames: Dict[str, List[dict]],
    grid_x: int,
    grid_y: int,
):
    """Return ordered freeze-frame records enriched with plotting context."""

    timeline: List[dict] = []
    for event in events:
        event_uuid = event.get("id")
        if not event_uuid:
            continue
        freeze_frame = freeze_frames.get(event_uuid)
        if not freeze_frame:
            continue

        context_grid = build_context_grid(
            freeze_frame, event.get("location"), grid_x, grid_y
        )
        teammates, opponents = extract_player_positions(freeze_frame)
        ball_location = event.get("location")
        ball_point = {
            "x": float(ball_location[0]) if ball_location else None,
            "y": float(ball_location[1]) if ball_location else None,
        }
        state_key, state_summary = summarise_state(
            freeze_frame, ball_location, grid_x, grid_y
        )

        timeline.append(
            {
                "event": event,
                "context": context_grid,
                "description": describe_event(event),
                "teammates": teammates,
                "opponents": opponents,
                "ball": ball_point,
                "state_key": state_key,
                "state_summary": state_summary,
                "action": infer_action(event),
            }
        )
    return timeline


def update_q_table_from_timeline(
    timeline: Sequence[dict],
    q_table: Dict[Tuple[int, ...], np.ndarray],
    q_gamma: float,
    q_alpha: float,
):
    """Apply tabular Q-learning updates for the provided timeline."""

    for idx, record in enumerate(timeline):
        next_record = timeline[idx + 1] if idx + 1 < len(timeline) else None
        reward = compute_reward(record["event"], next_record["event"] if next_record else None)

        actual_action = record["action"]
        if actual_action not in ACTION_SPACE:
            continue

        state_key = record["state_key"]
        q_values = q_table.setdefault(
            state_key, np.zeros(len(ACTION_SPACE), dtype=float)
        )

        next_max = 0.0
        if next_record is not None:
            next_state_key = next_record["state_key"]
            next_q = q_table.setdefault(
                next_state_key, np.zeros(len(ACTION_SPACE), dtype=float)
            )
            next_max = float(np.max(next_q))

        action_index = ACTION_SPACE.index(actual_action)
        target = reward + q_gamma * next_max
        q_values[action_index] += q_alpha * (target - q_values[action_index])


def pretrain_q_table(
    timelines: Sequence[Sequence[dict]],
    q_gamma: float,
    q_alpha: float,
    initial_q_table: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
):
    """Warm up the Q-table using the supplied timelines from prior matches."""

    if initial_q_table is not None:
        q_table = {
            state: values.copy() for state, values in initial_q_table.items()
        }
    else:
        q_table = {}

    for timeline in timelines:
        update_q_table_from_timeline(timeline, q_table, q_gamma, q_alpha)

    return q_table


def build_value_snapshots(
    timeline: Sequence[dict],
    grid_x: int,
    grid_y: int,
    gamma: float,
    alpha: float,
    q_gamma: float,
    q_alpha: float,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    seed: int,
    initial_q_table: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
) -> List[dict]:
    """Generate sequential value-grid snapshots for freeze-framed events."""

    value_grid = np.zeros((grid_y, grid_x), dtype=float)
    snapshots: List[dict] = []
    if initial_q_table is not None:
        q_table: Dict[Tuple[int, ...], np.ndarray] = {
            state: values.copy() for state, values in initial_q_table.items()
        }
    else:
        q_table = {}
    epsilon_value = epsilon
    rng = np.random.default_rng(seed)

    for idx, record in enumerate(timeline):
        next_record = timeline[idx + 1] if idx + 1 < len(timeline) else None
        reward = compute_reward(record["event"], next_record["event"] if next_record else None)

        context = record["context"]
        next_context = next_record["context"] if next_record else None

        state_value = float(np.sum(value_grid * context))
        next_value = float(np.sum(value_grid * next_context)) if next_context is not None else 0.0
        td_error = reward + gamma * next_value - state_value

        value_grid = value_grid + alpha * td_error * context

        state_key = record["state_key"]
        q_values = q_table.setdefault(state_key, np.zeros(len(ACTION_SPACE), dtype=float))
        if rng.random() < epsilon_value:
            suggested_idx = int(rng.integers(0, len(ACTION_SPACE)))
            explored = True
        else:
            suggested_idx = int(np.argmax(q_values))
            explored = False
        suggested_action = ACTION_SPACE[suggested_idx]

        next_state_key = next_record["state_key"] if next_record else None
        next_max = 0.0
        if next_state_key is not None:
            next_q = q_table.setdefault(next_state_key, np.zeros(len(ACTION_SPACE), dtype=float))
            next_max = float(np.max(next_q))

        actual_action = record["action"]
        if actual_action in ACTION_SPACE:
            action_index = ACTION_SPACE.index(actual_action)
            target = reward + q_gamma * next_max
            q_values[action_index] += q_alpha * (target - q_values[action_index])

        q_snapshot = {action: float(q_values[idx]) for idx, action in enumerate(ACTION_SPACE)}

        snapshots.append(
            {
                "value_grid": value_grid.copy(),
                "teammates": record["teammates"],
                "opponents": record["opponents"],
                "ball": record["ball"],
                "description": record["description"],
                "reward": reward,
                "state_value": state_value,
                "td_error": td_error,
                "suggested_action": suggested_action,
                "actual_action": actual_action,
                "exploration": explored,
                "epsilon": float(epsilon_value),
                "q_values": q_snapshot,
                "state_summary": record["state_summary"],
            }
        )
        epsilon_value = max(epsilon_min, epsilon_value * epsilon_decay)
    return snapshots


def pitch_shapes() -> List[dict]:
    """Return Plotly shape definitions for a stylised pitch."""

    shapes = [
        # Outer boundary
        {
            "type": "rect",
            "x0": 0,
            "y0": 0,
            "x1": PITCH_LENGTH,
            "y1": PITCH_WIDTH,
            "line": {"color": "white", "width": 2},
        },
        # Half-way line
        {
            "type": "line",
            "x0": PITCH_LENGTH / 2,
            "y0": 0,
            "x1": PITCH_LENGTH / 2,
            "y1": PITCH_WIDTH,
            "line": {"color": "white", "width": 1},
        },
        # Centre circle
        {
            "type": "circle",
            "xref": "x",
            "yref": "y",
            "x0": PITCH_LENGTH / 2 - 9.15,
            "y0": PITCH_WIDTH / 2 - 9.15,
            "x1": PITCH_LENGTH / 2 + 9.15,
            "y1": PITCH_WIDTH / 2 + 9.15,
            "line": {"color": "white", "width": 1},
        },
    ]

    # Penalty boxes and six-yard boxes
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
                "line": {"color": "white", "width": 1},
            },
            {
                "type": "rect",
                "x0": PITCH_LENGTH - penalty_length,
                "y0": (PITCH_WIDTH - penalty_width) / 2,
                "x1": PITCH_LENGTH,
                "y1": (PITCH_WIDTH + penalty_width) / 2,
                "line": {"color": "white", "width": 1},
            },
            {
                "type": "rect",
                "x0": 0,
                "y0": (PITCH_WIDTH - six_width) / 2,
                "x1": six_length,
                "y1": (PITCH_WIDTH + six_width) / 2,
                "line": {"color": "white", "width": 1},
            },
            {
                "type": "rect",
                "x0": PITCH_LENGTH - six_length,
                "y0": (PITCH_WIDTH - six_width) / 2,
                "x1": PITCH_LENGTH,
                "y1": (PITCH_WIDTH + six_width) / 2,
                "line": {"color": "white", "width": 1},
            },
        ]
    )
    return shapes


def make_animation(
    snapshots: List[dict],
    grid_x: int,
    grid_y: int,
    title: str,
    output_path: Path,
) -> None:
    """Create and save the interactive Plotly animation."""

    if not snapshots:
        raise RuntimeError("No freeze-frame events were found for this match.")

    x_centres = np.linspace(PITCH_LENGTH / (2 * grid_x), PITCH_LENGTH - PITCH_LENGTH / (2 * grid_x), grid_x)
    y_centres = np.linspace(PITCH_WIDTH / (2 * grid_y), PITCH_WIDTH - PITCH_WIDTH / (2 * grid_y), grid_y)

    z_min = min(float(np.min(frame["value_grid"])) for frame in snapshots)
    z_max = max(float(np.max(frame["value_grid"])) for frame in snapshots)
    max_abs = max(abs(z_min), abs(z_max))
    z_min, z_max = -max_abs, max_abs

    initial = snapshots[0]
    heatmap = go.Heatmap(
        z=initial["value_grid"],
        x=x_centres,
        y=y_centres,
        colorscale="RdBu",
        zmin=z_min,
        zmax=z_max,
        colorbar=dict(title="State value"),
        showscale=True,
    )

    teammate_trace = go.Scatter(
        x=[player["x"] for player in initial["teammates"]],
        y=[player["y"] for player in initial["teammates"]],
        mode="markers",
        marker=dict(size=10, color="#2ca02c"),
        name="Teammates",
        text=[player["name"] for player in initial["teammates"]],
        hovertemplate="%{text}<extra></extra>",
    )

    opponent_trace = go.Scatter(
        x=[player["x"] for player in initial["opponents"]],
        y=[player["y"] for player in initial["opponents"]],
        mode="markers",
        marker=dict(size=10, color="#d62728"),
        name="Opponents",
        text=[player["name"] for player in initial["opponents"]],
        hovertemplate="%{text}<extra></extra>",
    )

    ball_trace = go.Scatter(
        x=[initial["ball"]["x"]] if initial["ball"]["x"] is not None else [],
        y=[initial["ball"]["y"]] if initial["ball"]["y"] is not None else [],
        mode="markers",
        marker=dict(size=12, color="#ff7f0e", symbol="circle-open"),
        name="Ball",
        hoverinfo="skip",
    )

    frames = []
    slider_steps = []
    for idx, frame in enumerate(snapshots):
        description = frame["description"]
        nearest_distance = frame["state_summary"].get("nearest_opponent_distance")
        if nearest_distance is None:
            nearest_text = "Nearest opponent: not in frame"
        else:
            nearest_text = f"Nearest opponent: {nearest_distance:.1f} m"
        ball_cell = frame["state_summary"].get("ball_cell")
        behaviour = "explore" if frame["exploration"] else "exploit"
        actual_action = (
            frame["actual_action"].title() if frame["actual_action"] else "None"
        )
        suggested_action = frame["suggested_action"].title()
        summary_text = (
            f"{description}<br>"
            f"Reward: {frame['reward']:.2f}, TD error: {frame['td_error']:.2f}, "
            f"State value: {frame['state_value']:.2f}<br>"
            f"Agent suggestion: {suggested_action} ({behaviour}, ε={frame['epsilon']:.2f}) | "
            f"Actual: {actual_action}<br>"
            f"{nearest_text} | Ball cell: {ball_cell}"
        )
        q_text = "<br>".join(
            f"{action.title()}: {frame['q_values'][action]:.2f}" for action in ACTION_SPACE
        )
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=frame["value_grid"],
                        x=x_centres,
                        y=y_centres,
                        zmin=z_min,
                        zmax=z_max,
                        colorscale="RdBu",
                    ),
                    go.Scatter(
                        x=[player["x"] for player in frame["teammates"]],
                        y=[player["y"] for player in frame["teammates"]],
                        text=[player["name"] for player in frame["teammates"]],
                    ),
                    go.Scatter(
                        x=[player["x"] for player in frame["opponents"]],
                        y=[player["y"] for player in frame["opponents"]],
                        text=[player["name"] for player in frame["opponents"]],
                    ),
                    go.Scatter(
                        x=[frame["ball"]["x"]]
                        if frame["ball"]["x"] is not None
                        else [],
                        y=[frame["ball"]["y"]]
                        if frame["ball"]["y"] is not None
                        else [],
                    ),
                ],
                name=str(idx),
                layout=go.Layout(
                    annotations=[
                        dict(
                            x=0,
                            y=1.05,
                            xref="paper",
                            yref="paper",
                            text=summary_text,
                            showarrow=False,
                            font=dict(color="white"),
                        ),
                        dict(
                            x=1.12,
                            y=0.5,
                            xref="paper",
                            yref="paper",
                            text=f"Q-values<br>{q_text}",
                            showarrow=False,
                            align="left",
                            bgcolor="rgba(0,0,0,0.6)",
                            font=dict(color="white"),
                        ),
                    ]
                ),
            )
        )
        slider_steps.append(
            {
                "args": [[str(idx)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(idx + 1),
                "method": "animate",
            }
        )

    fig = go.Figure(data=[heatmap, teammate_trace, opponent_trace, ball_trace], frames=frames)
    initial_nearest = initial["state_summary"].get("nearest_opponent_distance")
    if initial_nearest is None:
        initial_nearest_text = "Nearest opponent: not in frame"
    else:
        initial_nearest_text = f"Nearest opponent: {initial_nearest:.1f} m"
    initial_summary = (
        f"{initial['description']}<br>"
        f"Reward: {initial['reward']:.2f}, TD error: {initial['td_error']:.2f}, "
        f"State value: {initial['state_value']:.2f}<br>"
        f"Agent suggestion: {initial['suggested_action'].title()} "
        f"({'explore' if initial['exploration'] else 'exploit'}, ε={initial['epsilon']:.2f}) | "
        f"Actual: {initial['actual_action'].title() if initial['actual_action'] else 'None'}<br>"
        f"{initial_nearest_text} | Ball cell: {initial['state_summary'].get('ball_cell')}"
    )
    initial_q_text = "<br>".join(
        f"{action.title()}: {initial['q_values'][action]:.2f}" for action in ACTION_SPACE
    )
    fig.update_layout(
        title=title,
        width=1150,
        height=650,
        paper_bgcolor="#1b1b1b",
        plot_bgcolor="#1b1b1b",
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
        shapes=pitch_shapes(),
        annotations=[
            dict(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text=initial_summary,
                showarrow=False,
                font=dict(color="white"),
            ),
            dict(
                x=1.12,
                y=0.5,
                xref="paper",
                yref="paper",
                text=f"Q-values<br>{initial_q_text}",
                showarrow=False,
                align="left",
                bgcolor="rgba(0,0,0,0.6)",
                font=dict(color="white"),
            ),
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Event: ", "font": {"color": "white"}},
                "pad": {"t": 50, "b": 10},
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
                        "args": [None, {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 50},
                "x": 0.1,
                "y": 0,
                "showactive": False,
                "bgcolor": "#333333",
                "bordercolor": "#444444",
            }
        ],
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        margin=dict(l=60, r=240, t=80, b=40),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_play=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise a StatsBomb 360 Markov value model with interactive playback.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        help="Limit to a specific competition ID when searching for matches.",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        help="Limit to a specific season ID when searching for matches.",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        help="Use a specific match_id (must have StatsBomb 360 data).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=5,
        help="Maximum matches to scan when searching for 360 data.",
    )
    parser.add_argument(
        "--grid-x",
        type=int,
        default=12,
        help="Number of grid columns for the value function.",
    )
    parser.add_argument(
        "--grid-y",
        type=int,
        default=8,
        help="Number of grid rows for the value function.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for the temporal-difference update.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Learning rate for the temporal-difference update.",
    )
    parser.add_argument(
        "--q-gamma",
        type=float,
        default=0.9,
        help="Discount factor for the Q-learning policy update.",
    )
    parser.add_argument(
        "--q-alpha",
        type=float,
        default=0.3,
        help="Learning rate applied to the Q-learning updates.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.25,
        help="Initial exploration rate for the epsilon-greedy policy.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative decay applied to epsilon after each frame.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Lower bound for exploration to retain some stochasticity.",
    )
    parser.add_argument(
        "--rl-seed",
        type=int,
        default=42,
        help="Random seed used for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--pretrain-count",
        type=int,
        default=0,
        help="Number of additional matches used to warm up the Q-table before the replay.",
    )
    parser.add_argument(
        "--pretrain-match-id",
        type=int,
        action="append",
        help="Explicit match_id values to pretrain on (can be supplied multiple times).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("three_sixty_markov.html"),
        help="Output HTML file for the interactive animation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered matches with 360 data and exit.",
    )
    return parser.parse_args()


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
                f"{summary.home_team} vs {summary.away_team} | "
                f"{summary.competition_name} {summary.season_name} | "
                f"freeze frames: {summary.freeze_frame_events}"
            )
        return

    if args.match_id is not None:
        target_match = resolve_match_summary(args.match_id, matches)
    else:
        if not matches:
            raise RuntimeError(
                "No matches with StatsBomb 360 data were found. Try increasing --max-matches "
                "or adjusting the competition/season filters."
            )
        target_match = matches[0]

    pretrain_matches: Dict[int, MatchSummary] = {}
    if args.pretrain_match_id:
        for match_id in args.pretrain_match_id:
            if match_id == target_match.match_id:
                continue
            if match_id in pretrain_matches:
                continue
            pretrain_matches[match_id] = resolve_match_summary(match_id, matches)

    if args.pretrain_count > 0:
        for summary in matches:
            if summary.match_id == target_match.match_id:
                continue
            if summary.match_id in pretrain_matches:
                continue
            pretrain_matches[summary.match_id] = summary
            if len(pretrain_matches) >= args.pretrain_count:
                break
        if len(pretrain_matches) < args.pretrain_count:
            print(
                "Warning: fewer matches were available for pretraining than requested. "
                "Increase --max-matches or supply explicit --pretrain-match-id values."
            )

    pretrain_timelines: List[List[dict]] = []
    if pretrain_matches:
        print(
            f"Pretraining Q-table on {len(pretrain_matches)} matches before visualising the target replay."
        )
    for match_id, summary in pretrain_matches.items():
        freeze_frames = load_freeze_frames(match_id)
        if not freeze_frames:
            print(f"Skipping match {match_id}: StatsBomb 360 freeze frames are not available.")
            continue
        events = load_events(match_id)
        timeline = construct_timeline(events, freeze_frames, args.grid_x, args.grid_y)
        if not timeline:
            print(f"Skipping match {match_id}: no usable freeze frames found.")
            continue
        pretrain_timelines.append(timeline)

    pretrained_q: Optional[Dict[Tuple[int, ...], np.ndarray]] = None
    if pretrain_timelines:
        pretrained_q = pretrain_q_table(
            pretrain_timelines,
            q_gamma=args.q_gamma,
            q_alpha=args.q_alpha,
        )

    freeze_frames = load_freeze_frames(target_match.match_id)
    if not freeze_frames:
        raise RuntimeError(
            f"Match {target_match.match_id} no longer exposes StatsBomb 360 data."
        )

    events = load_events(target_match.match_id)
    timeline = construct_timeline(events, freeze_frames, args.grid_x, args.grid_y)
    snapshots = build_value_snapshots(
        timeline,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        gamma=args.gamma,
        alpha=args.alpha,
        q_gamma=args.q_gamma,
        q_alpha=args.q_alpha,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.rl_seed,
        initial_q_table=pretrained_q,
    )

    title = (
        f"{target_match.home_team} vs {target_match.away_team} ({target_match.match_date})<br>"
        f"{target_match.competition_name} {target_match.season_name}"
    )
    make_animation(snapshots, args.grid_x, args.grid_y, title, args.output)
    print(f"Interactive animation saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
