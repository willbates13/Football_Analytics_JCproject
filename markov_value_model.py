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


def build_value_snapshots(
    events: List[dict],
    freeze_frames: Dict[str, List[dict]],
    grid_x: int,
    grid_y: int,
    gamma: float,
    alpha: float,
) -> List[dict]:
    """Generate sequential value-grid snapshots for freeze-framed events."""

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

        timeline.append(
            {
                "event": event,
                "context": context_grid,
                "description": describe_event(event),
                "teammates": teammates,
                "opponents": opponents,
                "ball": ball_point,
            }
        )

    value_grid = np.zeros((grid_y, grid_x), dtype=float)
    snapshots: List[dict] = []

    for idx, record in enumerate(timeline):
        next_record = timeline[idx + 1] if idx + 1 < len(timeline) else None
        reward = compute_reward(record["event"], next_record["event"] if next_record else None)

        context = record["context"]
        next_context = next_record["context"] if next_record else None

        state_value = float(np.sum(value_grid * context))
        next_value = float(np.sum(value_grid * next_context)) if next_context is not None else 0.0
        td_error = reward + gamma * next_value - state_value

        value_grid = value_grid + alpha * td_error * context

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
            }
        )
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
                            text=f"{description}<br>Reward: {frame['reward']:.2f}, TD error: {frame['td_error']:.2f}",
                            showarrow=False,
                            font=dict(color="white"),
                        )
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
    fig.update_layout(
        title=title,
        width=1000,
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
                text=f"{initial['description']}<br>Reward: {initial['reward']:.2f}, TD error: {initial['td_error']:.2f}",
                showarrow=False,
                font=dict(color="white"),
            )
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

    target_match: Optional[MatchSummary] = None
    if args.match_id is not None:
        for summary in matches:
            if summary.match_id == args.match_id:
                target_match = summary
                break
        if target_match is None:
            # If the requested match wasn't in the limited search, fetch directly.
            freeze_frames = load_freeze_frames(args.match_id)
            if not freeze_frames:
                raise RuntimeError(
                    f"Match {args.match_id} does not have StatsBomb 360 data available."
                )
            # We need metadata to label the animation; attempt to locate it.
            for competition in load_competitions():
                comp_id = int(competition["competition_id"])
                season_id = int(competition["season_id"])
                matches_meta = load_matches(comp_id, season_id)
                for match in matches_meta:
                    if int(match["match_id"]) == args.match_id:
                        target_match = MatchSummary(
                            competition_id=comp_id,
                            competition_name=competition.get("competition_name", ""),
                            season_id=season_id,
                            season_name=competition.get("season_name", ""),
                            match_id=args.match_id,
                            home_team=match.get("home_team", {}).get("home_team_name", ""),
                            away_team=match.get("away_team", {}).get("away_team_name", ""),
                            match_date=match.get("match_date", ""),
                            freeze_frame_events=sum(
                                1 for frame in freeze_frames.values() if frame
                            ),
                        )
                        break
                if target_match:
                    break
            if target_match is None:
                raise RuntimeError(
                    "Unable to locate metadata for the requested match, "
                    "even though 360 data exists."
                )
    else:
        if not matches:
            raise RuntimeError(
                "No matches with StatsBomb 360 data were found. Try increasing --max-matches "
                "or adjusting the competition/season filters."
            )
        target_match = matches[0]

    assert target_match is not None
    freeze_frames = load_freeze_frames(target_match.match_id)
    if not freeze_frames:
        raise RuntimeError(
            f"Match {target_match.match_id} no longer exposes StatsBomb 360 data."
        )

    events = load_events(target_match.match_id)
    snapshots = build_value_snapshots(
        events,
        freeze_frames,
        grid_x=args.grid_x,
        grid_y=args.grid_y,
        gamma=args.gamma,
        alpha=args.alpha,
    )

    title = (
        f"{target_match.home_team} vs {target_match.away_team} ({target_match.match_date})<br>"
        f"{target_match.competition_name} {target_match.season_name}"
    )
    make_animation(snapshots, args.grid_x, args.grid_y, title, args.output)
    print(f"Interactive animation saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
