"""Enhanced Markov chain value model for StatsBomb open data.

This script downloads a sample of StatsBomb open-data matches, builds a
possession-based Markov chain that approximates Ian Graham's possession value
model, and produces a suite of visualisations:

* an interactive Plotly heatmap with higher-resolution grids and goal markers,
* a static Matplotlib figure for presentations,
* a support-aware model using StatsBomb 360 freeze frames,
* an average ball-progression quiver plot, and
* a shot quality heatmap by distance and angle.

Usage (from the repository root):

    python markov_value_model.py --max-matches 20

The script keeps the file structure intentionally simple so it can be dropped
into presentations or notebooks without additional project scaffolding.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
import numpy as np
import plotly.graph_objects as go
import requests
from requests.exceptions import HTTPError

STATS_BOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Pitch and grid settings (StatsBomb pitch: 120 x 80 coordinates)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_X = 12
GRID_Y = 8
CELL_LENGTH = PITCH_LENGTH / GRID_X
CELL_WIDTH = PITCH_WIDTH / GRID_Y


def fetch_json(url: str) -> list:
    """Download JSON data from StatsBomb's open-data GitHub repository."""

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_matches(competition_id: int, season_id: int) -> list:
    """Load the list of matches for a competition/season pair."""

    url = f"{STATS_BOMB_BASE}/matches/{competition_id}/{season_id}.json"
    return fetch_json(url)


def load_match_events(match_id: int) -> list:
    """Fetch the events for a specific match."""

    url = f"{STATS_BOMB_BASE}/events/{match_id}.json"
    return fetch_json(url)


def load_three_sixty(match_id: int) -> Dict[str, list]:
    """Fetch StatsBomb 360 freeze frames for a match (if available)."""

    url = f"{STATS_BOMB_BASE}/three-sixty/{match_id}.json"
    try:
        frames = fetch_json(url)
    except HTTPError:
        return {}

    frame_map = {}
    for frame in frames:
        event_uuid = frame.get("event_uuid")
        if event_uuid:
            frame_map[event_uuid] = frame.get("freeze_frame", [])
    return frame_map


def location_to_zone(location: Sequence[float]) -> Optional[int]:
    """Convert a StatsBomb (x, y) location into a grid zone index."""

    if not location or len(location) < 2:
        return None

    x, y = location[:2]
    if x is None or y is None:
        return None

    # Clamp to pitch boundaries just in case the feed contains out-of-bounds
    x = min(max(x, 0.0), PITCH_LENGTH - 1e-6)
    y = min(max(y, 0.0), PITCH_WIDTH - 1e-6)

    x_bin = int(x / (PITCH_LENGTH / GRID_X))
    y_bin = int(y / (PITCH_WIDTH / GRID_Y))
    return y_bin * GRID_X + x_bin


def zone_to_grid(zone: int) -> Tuple[int, int]:
    """Return the (x, y) grid coordinate for a zone index."""

    x = zone % GRID_X
    y = zone // GRID_X
    return x, y


def find_next_possession_event(events: List[dict], start_idx: int, possession_id: int) -> Optional[dict]:
    """Return the next event in the same possession with a location."""

    for event in events[start_idx + 1 :]:
        if event.get("possession") != possession_id:
            return None
        if location_to_zone(event.get("location")) is not None:
            return event
        # Continue searching within the same possession for an actionable event
    return None


def determine_next_state(event: dict, next_event: Optional[dict]) -> Tuple[str, Optional[int]]:
    """Determine the transition outcome for an event.

    Returns a tuple of (state_type, value) where state_type is one of:
    - "zone": possession continues in another zone.
    - "goal": the action results in a goal for the possessing team.
    - "turnover": the possession ends without a goal.
    """

    if "shot" in event:
        outcome = event["shot"].get("outcome", {}).get("name")
        if outcome == "Goal":
            return "goal", None
        return "turnover", None

    if next_event is None:
        return "turnover", None

    next_zone = location_to_zone(next_event.get("location"))
    if next_zone is None:
        return "turnover", None

    return "zone", next_zone


def build_transition_counts(matches_events: Iterable[List[dict]]) -> Tuple[Dict[int, Counter], Dict[Tuple[int, str], Counter]]:
    """Aggregate transition counts for states and actions."""

    zone_counts: Dict[int, Counter] = defaultdict(Counter)
    action_counts: Dict[Tuple[int, str], Counter] = defaultdict(Counter)

    for events in matches_events:
        for idx, event in enumerate(events):
            zone = location_to_zone(event.get("location"))
            if zone is None:
                continue

            action = event.get("type", {}).get("name", "Unknown")
            possession_id = event.get("possession")
            next_event = find_next_possession_event(events, idx, possession_id)
            state_type, state_value = determine_next_state(event, next_event)

            zone_counts[zone][state_type if state_value is None else (state_type, state_value)] += 1
            action_counts[(zone, action)][state_type if state_value is None else (state_type, state_value)] += 1

    return zone_counts, action_counts


def counts_to_transition_matrix(zone_counts: Dict[int, Counter]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert transition counters into matrices.

    Returns:
        P (np.ndarray): transition matrix between in-play zones.
        goal_probs (np.ndarray): probability of scoring on the next action.
        turnover_probs (np.ndarray): probability of ending the possession.
    """

    num_zones = GRID_X * GRID_Y
    P = np.zeros((num_zones, num_zones))
    goal_probs = np.zeros(num_zones)
    turnover_probs = np.zeros(num_zones)

    for zone in range(num_zones):
        counts = zone_counts.get(zone, Counter())
        total = sum(counts.values())
        if total == 0:
            continue

        for outcome, count in counts.items():
            if outcome == "goal":
                goal_probs[zone] += count / total
            elif outcome == "turnover":
                turnover_probs[zone] += count / total
            else:
                _, next_zone = outcome
                P[zone, next_zone] += count / total

        # Normalise any rounding errors
        row_sum = P[zone].sum() + goal_probs[zone] + turnover_probs[zone]
        if row_sum > 0:
            P[zone] /= row_sum
            goal_probs[zone] /= row_sum
            turnover_probs[zone] /= row_sum

    return P, goal_probs, turnover_probs


def solve_state_value(P: np.ndarray, goal_probs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Solve the Markov chain value function.

    Args:
        P: Transition probabilities between non-terminal states.
        goal_probs: Probability of scoring immediately from each state.
        gamma: Discount factor. A value < 1 emphasises immediate rewards.
    """

    identity = np.eye(P.shape[0])
    system_matrix = identity - gamma * P
    return np.linalg.solve(system_matrix, goal_probs)


def compute_action_values(
    action_counts: Dict[Tuple[int, str], Counter],
    state_values: np.ndarray,
) -> Dict[Tuple[int, str], float]:
    """Calculate the expected value of taking each action in each zone."""

    action_values: Dict[Tuple[int, str], float] = {}

    for (zone, action), counts in action_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue

        value = 0.0
        for outcome, count in counts.items():
            prob = count / total
            if outcome == "goal":
                value += prob
            elif outcome == "turnover":
                continue
            else:
                _, next_zone = outcome
                value += prob * state_values[next_zone]

        action_values[(zone, action)] = value

    return action_values


def grid_from_values(values: Dict[int, float], default: Optional[float] = None) -> np.ndarray:
    """Map zone-indexed values onto a 2D grid for plotting."""

    grid = np.full((GRID_Y, GRID_X), np.nan)
    for zone, val in values.items():
        x, y = zone_to_grid(zone)
        # Flip y-axis so the attacking goal (x = 120) appears at the top of the heatmap
        grid[GRID_Y - 1 - y, x] = val

    if default is not None:
        grid = np.nan_to_num(grid, nan=default)

    return grid


def grid_centres() -> Tuple[np.ndarray, np.ndarray]:
    """Return the x/y coordinates for the centre of each zone cell."""

    x_centers = np.linspace(CELL_LENGTH / 2, PITCH_LENGTH - CELL_LENGTH / 2, GRID_X)
    y_centers = np.linspace(PITCH_WIDTH - CELL_WIDTH / 2, CELL_WIDTH / 2, GRID_Y)
    return x_centers, y_centers


def create_pitch_shapes() -> List[dict]:
    """Return Plotly shape definitions for a simple pitch with goals."""

    penalty_width = 44.0
    six_yard_width = 20.0
    goal_width = 8.0
    penalty_depth = 18.0
    six_yard_depth = 6.0
    goal_depth = 2.0
    center_y = PITCH_WIDTH / 2

    shapes: List[dict] = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_LENGTH, y1=PITCH_WIDTH, line=dict(color="black", width=2)),
        dict(type="line", x0=PITCH_LENGTH / 2, y0=0, x1=PITCH_LENGTH / 2, y1=PITCH_WIDTH, line=dict(color="black", width=1)),
        dict(
            type="circle",
            x0=PITCH_LENGTH / 2 - 10,
            y0=center_y - 10,
            x1=PITCH_LENGTH / 2 + 10,
            y1=center_y + 10,
            line=dict(color="black", width=1),
        ),
    ]

    for side in (0, PITCH_LENGTH - penalty_depth):
        direction = 1 if side == 0 else -1
        x0 = side
        x1 = side + direction * penalty_depth
        shapes.append(
            dict(
                type="rect",
                x0=min(x0, x1),
                x1=max(x0, x1),
                y0=center_y - penalty_width / 2,
                y1=center_y + penalty_width / 2,
                line=dict(color="black", width=1),
            )
        )
        x0 = side
        x1 = side + direction * six_yard_depth
        shapes.append(
            dict(
                type="rect",
                x0=min(x0, x1),
                x1=max(x0, x1),
                y0=center_y - six_yard_width / 2,
                y1=center_y + six_yard_width / 2,
                line=dict(color="black", width=1),
            )
        )
        goal_x0 = side - direction * goal_depth
        goal_x1 = side
        shapes.append(
            dict(
                type="rect",
                x0=min(goal_x0, goal_x1),
                x1=max(goal_x0, goal_x1),
                y0=center_y - goal_width / 2,
                y1=center_y + goal_width / 2,
                line=dict(color="black", width=2),
                fillcolor="rgba(200,200,200,0.3)",
            )
        )

    return shapes


def build_interactive_figure(
    state_values: np.ndarray,
    discounted_state_values: np.ndarray,
    action_values: Dict[Tuple[int, str], float],
) -> go.Figure:
    """Create a Plotly figure with interactive dropdowns for different metrics."""

    num_zones = GRID_X * GRID_Y
    zone_value_map = {zone: state_values[zone] for zone in range(num_zones)}
    discounted_map = {zone: discounted_state_values[zone] for zone in range(num_zones)}

    heatmaps: List[Dict[str, object]] = []

    def append_heatmap(matrix: np.ndarray, label: str, *, diverging: bool = False, colorbar: str = "xG") -> None:
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            zmin, zmax = (0.0, 1.0)
        else:
            zmin, zmax = float(finite.min()), float(finite.max())
        colorscale = "YlOrRd"
        if diverging:
            bound = max(abs(zmin), abs(zmax)) or 1.0
            zmin, zmax = -bound, bound
            colorscale = "RdBu"
        elif zmin == zmax:
            zmax = zmin + 1e-6

        heatmaps.append(
            {
                "matrix": matrix,
                "label": label,
                "zmin": zmin,
                "zmax": zmax,
                "colorscale": colorscale,
                "colorbar": colorbar,
            }
        )

    append_heatmap(grid_from_values(zone_value_map), "State Value (xG probability)")
    append_heatmap(grid_from_values(discounted_map), "Discounted Value (quick chance focus)")

    base_values = state_values.copy()
    actions_by_name = sorted({action for _, action in action_values.keys()})
    for action in actions_by_name:
        action_map = {}
        advantage_map = {}
        for zone in range(num_zones):
            key = (zone, action)
            if key in action_values:
                action_map[zone] = action_values[key]
                advantage_map[zone] = action_values[key] - base_values[zone]

        if action_map:
            append_heatmap(grid_from_values(action_map), f"Action Value: {action}")
        if advantage_map:
            append_heatmap(
                grid_from_values(advantage_map),
                f"Action Advantage: {action}",
                diverging=True,
                colorbar="Advantage",
            )

    x_centers, y_centers = grid_centres()

    traces = []
    for spec in heatmaps:
        traces.append(
            go.Heatmap(
                z=spec["matrix"],
                x=x_centers,
                y=y_centers,
                colorscale=spec["colorscale"],
                zmin=spec["zmin"],
                zmax=spec["zmax"],
                colorbar=dict(title=spec["colorbar"]),
                hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>value: %{z:.3f}<extra></extra>",
                visible=False,
            )
        )

    if traces:
        traces[0].visible = True

    fig = go.Figure(data=traces)

    buttons = []
    for idx, spec in enumerate(heatmaps):
        visibility = [False] * len(traces)
        visibility[idx] = True
        buttons.append(
            dict(label=spec["label"], method="update", args=[{"visible": visibility}, {"title": spec["label"]}])
        )

    fig.update_layout(
        title=heatmaps[0]["label"] if heatmaps else "Markov Chain Value Model",
        xaxis=dict(title="Pitch length (yards)", range=[0, PITCH_LENGTH], constrain="domain"),
        yaxis=dict(title="Pitch width (yards)", range=[PITCH_WIDTH, 0], scaleanchor="x", scaleratio=1),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.05, xanchor="left")],
        template="plotly_white",
        shapes=create_pitch_shapes(),
        width=950,
        height=600,
    )

    return fig


def draw_pitch(ax: "Axes") -> None:
    """Render a simple football pitch with goals onto a Matplotlib axis."""

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_aspect("equal")
    ax.axis("off")

    # Outer lines and halfway line
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color="black", linewidth=1.5)
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color="black", linewidth=1)

    center_circle = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 10, fill=False, color="black", linewidth=1)
    ax.add_patch(center_circle)

    penalty_width = 44.0
    penalty_depth = 18.0
    six_width = 20.0
    six_depth = 6.0
    goal_width = 8.0
    goal_depth = 2.0
    center_y = PITCH_WIDTH / 2

    for side in (0, PITCH_LENGTH):
        direction = 1 if side == 0 else -1
        penalty = plt.Rectangle(
            (side - (1 - direction) * penalty_depth, center_y - penalty_width / 2),
            penalty_depth * direction,
            penalty_width,
            fill=False,
            color="black",
            linewidth=1,
        )
        six = plt.Rectangle(
            (side - (1 - direction) * six_depth, center_y - six_width / 2),
            six_depth * direction,
            six_width,
            fill=False,
            color="black",
            linewidth=1,
        )
        goal = plt.Rectangle(
            (side - (1 - direction) * goal_depth, center_y - goal_width / 2),
            goal_depth * direction,
            goal_width,
            fill=True,
            facecolor="lightgrey",
            edgecolor="black",
            linewidth=1,
            alpha=0.4,
        )
        ax.add_patch(penalty)
        ax.add_patch(six)
        ax.add_patch(goal)


def render_static_heatmap(ax: "Axes", matrix: np.ndarray, title: str, *, cmap: str = "YlOrRd", diverging: bool = False) -> None:
    """Render a heatmap atop the pitch background."""

    draw_pitch(ax)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        vmin, vmax = (0.0, 1.0)
    else:
        vmin, vmax = float(finite.min()), float(finite.max())

    if diverging:
        bound = max(abs(vmin), abs(vmax)) or 1.0
        vmin, vmax = -bound, bound
        cmap = "RdBu_r"
    elif vmin == vmax:
        vmax = vmin + 1e-6

    ax.imshow(
        np.ma.masked_invalid(matrix),
        extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
    )
    ax.set_title(title)


def create_static_value_figure(
    state_values: np.ndarray,
    discounted_state_values: np.ndarray,
    action_values: Dict[Tuple[int, str], float],
    output: Path,
) -> None:
    """Save a Matplotlib figure highlighting key value surfaces."""

    num_zones = GRID_X * GRID_Y
    base_map = {zone: state_values[zone] for zone in range(num_zones)}
    discounted_map = {zone: discounted_state_values[zone] for zone in range(num_zones)}

    # Choose the two most common actions by coverage
    action_coverage = Counter()
    for (zone, action), value in action_values.items():
        if value:
            action_coverage[action] += 1

    most_common = [action for action, _ in action_coverage.most_common(2)]
    action_grids = []
    for action in most_common:
        action_map = {zone: value for (zone, name), value in action_values.items() if name == action}
        action_grids.append((action, grid_from_values(action_map)))

    if plt is None:
        raise RuntimeError("Matplotlib is required for static output")

    fig, axes = plt.subplots(1, 2 + len(action_grids), figsize=(5 * (2 + len(action_grids)), 6))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    render_static_heatmap(axes[0], grid_from_values(base_map), "State value")
    render_static_heatmap(
        axes[1],
        grid_from_values(discounted_map),
        "Discounted value",
    )

    for idx, (action, matrix) in enumerate(action_grids, start=2):
        render_static_heatmap(axes[idx], matrix, f"Action value: {action}")

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def possession_goal_outcome(events: List[dict], start_idx: int, possession_id: int, team_id: int) -> bool:
    """Return True if the possession results in a goal for the given team."""

    for event in events[start_idx:]:
        if event.get("possession") != possession_id:
            break
        if event.get("team", {}).get("id") != team_id:
            continue
        shot = event.get("shot")
        if shot and shot.get("outcome", {}).get("name") == "Goal":
            return True
    return False


def freeze_frame_support_label(freeze_frame: List[dict], location: Sequence[float]) -> str:
    """Encode supporting player context into a categorical label."""

    if not location or len(location) < 2:
        return "unknown"

    teammates_near = 0
    opponents_near = 0
    teammates_ahead = 0

    for frame in freeze_frame:
        point = frame.get("location")
        if not point or len(point) < 2:
            continue
        dx = point[0] - location[0]
        dy = point[1] - location[1]
        distance = float(np.hypot(dx, dy))
        if frame.get("teammate"):
            if distance <= 15:
                teammates_near += 1
            if dx > 0:
                teammates_ahead += 1
        else:
            if distance <= 10:
                opponents_near += 1

    if teammates_near == 0:
        support = "isolated"
    elif teammates_near <= 2:
        support = "supported"
    else:
        support = "overload"

    if opponents_near == 0:
        pressure = "free"
    elif opponents_near == 1:
        pressure = "pressure"
    else:
        pressure = "crowded"

    ahead = "no_lane" if teammates_ahead == 0 else "lane"
    return f"{support}/{pressure}/{ahead}"


def compute_support_state_values(
    matches_events: Iterable[Tuple[List[dict], Dict[str, List[dict]]]]
) -> Dict[str, Dict[int, float]]:
    """Estimate scoring probability conditioned on support context states."""

    totals: Dict[Tuple[str, int], Counter] = defaultdict(Counter)

    for events, frame_map in matches_events:
        event_index = {event.get("id"): idx for idx, event in enumerate(events)}
        for event_id, freeze_frame in frame_map.items():
            idx = event_index.get(event_id)
            if idx is None:
                continue

            event = events[idx]
            zone = location_to_zone(event.get("location"))
            if zone is None:
                continue

            team = event.get("team", {}).get("id")
            possession_id = event.get("possession")
            if team is None or possession_id is None:
                continue

            label = freeze_frame_support_label(freeze_frame, event.get("location", []))
            goal = possession_goal_outcome(events, idx, possession_id, team)
            totals[(label, zone)]["count"] += 1
            if goal:
                totals[(label, zone)]["goals"] += 1

    support_maps: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (label, zone), counts in totals.items():
        count = counts["count"]
        goals = counts["goals"]
        if count > 0:
            support_maps[label][zone] = goals / count

    return dict(support_maps)


def build_support_figure(support_maps: Dict[str, Dict[int, float]]) -> go.Figure:
    """Create a dropdown heatmap for the support-aware state values."""

    x_centers, y_centers = grid_centres()
    heatmaps = []
    for label, zone_map in sorted(support_maps.items(), key=lambda item: item[0]):
        matrix = grid_from_values(zone_map, default=np.nan)
        finite = matrix[np.isfinite(matrix)]
        if finite.size == 0:
            continue
        heatmaps.append((label, matrix, float(finite.min()), float(finite.max())))

    traces = []
    for label, matrix, vmin, vmax in heatmaps:
        traces.append(
            go.Heatmap(
                z=matrix,
                x=x_centers,
                y=y_centers,
                colorscale="Viridis",
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(title="xG"),
                hovertemplate="x: %{x:.1f}<br>y: %{y:.1f}<br>value: %{z:.3f}<extra></extra>",
                visible=False,
            )
        )

    if traces:
        traces[0].visible = True

    buttons = []
    for idx, (label, _, _, _) in enumerate(heatmaps):
        visibility = [False] * len(traces)
        visibility[idx] = True
        buttons.append(dict(label=label, method="update", args=[{"visible": visibility}, {"title": label}]))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=heatmaps[0][0] if heatmaps else "Support context value",
        xaxis=dict(title="Pitch length (yards)", range=[0, PITCH_LENGTH], constrain="domain"),
        yaxis=dict(title="Pitch width (yards)", range=[PITCH_WIDTH, 0], scaleanchor="x", scaleratio=1),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True, x=1.05, xanchor="left")],
        template="plotly_white",
        shapes=create_pitch_shapes(),
        width=950,
        height=600,
    )
    return fig


def compute_ball_progression_vectors(matches_events: Iterable[List[dict]]) -> Dict[int, Tuple[float, float, int]]:
    """Compute average ball movement vectors (dx, dy, count) from each zone."""

    vectors: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    for events in matches_events:
        for event in events:
            zone = location_to_zone(event.get("location"))
            if zone is None:
                continue

            end_location: Optional[Sequence[float]] = None
            if "pass" in event:
                end_location = event["pass"].get("end_location")
            elif "carry" in event:
                end_location = event["carry"].get("end_location")

            if end_location and len(end_location) >= 2:
                dx = end_location[0] - event["location"][0]
                dy = end_location[1] - event["location"][1]
                vectors[zone].append((dx, dy))

    aggregated: Dict[int, Tuple[float, float, int]] = {}
    for zone, displacements in vectors.items():
        if displacements:
            dxs, dys = zip(*displacements)
            aggregated[zone] = (float(np.mean(dxs)), float(np.mean(dys)), len(displacements))

    return aggregated


def create_progression_plot(vectors: Dict[int, Tuple[float, float, int]], output: Path) -> None:
    """Render a quiver plot showing average ball progression from each zone."""

    if plt is None:
        raise RuntimeError("Matplotlib is required for the progression plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax)

    x_centers, y_centers = grid_centres()
    X, Y = np.meshgrid(x_centers, y_centers)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for zone, (dx, dy, count) in vectors.items():
        x_idx, y_idx = zone_to_grid(zone)
        row = GRID_Y - 1 - y_idx
        col = x_idx
        U[row, col] = dx
        V[row, col] = -dy  # invert y-axis for plotting

    magnitude = np.hypot(U, V)
    max_mag = np.nanmax(magnitude) if np.isfinite(np.nanmax(magnitude)) else 1.0
    scale = max_mag if max_mag else 1.0

    quiver = ax.quiver(X, Y, U, V, magnitude, angles="xy", scale_units="xy", scale=1.5 * scale, cmap="plasma")
    cbar = fig.colorbar(quiver, ax=ax)
    cbar.set_label("Average movement (yards)")
    ax.set_title("Average ball progression for passes and carries")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def compute_shot_quality(events: Iterable[List[dict]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute goal probability by shot distance and angle bins."""

    distances = []
    angles = []
    outcomes = []

    goal_center = np.array([PITCH_LENGTH, PITCH_WIDTH / 2])

    for match_events in events:
        for event in match_events:
            shot = event.get("shot")
            if not shot:
                continue
            location = event.get("location")
            if not location or len(location) < 2:
                continue
            point = np.array(location[:2])
            vector = goal_center - point
            distance = float(np.hypot(vector[0], vector[1]))
            angle = float(np.degrees(np.arctan2(abs(vector[1]), vector[0])))
            outcome = 1.0 if shot.get("outcome", {}).get("name") == "Goal" else 0.0
            distances.append(distance)
            angles.append(angle)
            outcomes.append(outcome)

    if not distances:
        return np.array([]), np.array([]), np.array([[]])

    dist_bins = np.linspace(0, 40, 17)
    angle_bins = np.linspace(0, 90, 19)

    counts = np.zeros((len(angle_bins) - 1, len(dist_bins) - 1))
    goals = np.zeros_like(counts)

    for d, a, o in zip(distances, angles, outcomes):
        dist_idx = np.searchsorted(dist_bins, d, side="right") - 1
        angle_idx = np.searchsorted(angle_bins, a, side="right") - 1
        if 0 <= dist_idx < counts.shape[1] and 0 <= angle_idx < counts.shape[0]:
            counts[angle_idx, dist_idx] += 1
            goals[angle_idx, dist_idx] += o

    with np.errstate(divide="ignore", invalid="ignore"):
        probabilities = np.divide(goals, counts, out=np.zeros_like(goals), where=counts > 0)

    return dist_bins, angle_bins, probabilities


def build_shot_quality_figure(dist_bins: np.ndarray, angle_bins: np.ndarray, probabilities: np.ndarray) -> go.Figure:
    """Create a heatmap describing which shot profiles score most often."""

    if probabilities.size == 0:
        return go.Figure()

    dist_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=probabilities,
                x=dist_centers,
                y=angle_centers,
                colorscale="Turbo",
                colorbar=dict(title="Goal probability"),
                hovertemplate="Distance: %{x:.1f} yd<br>Angle: %{y:.1f}°<br>Goals: %{z:.3f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Shot quality by distance and angle",
        xaxis=dict(title="Distance from goal (yards)"),
        yaxis=dict(title="Angle to goal (degrees)"),
        template="plotly_white",
        width=800,
        height=550,
    )

    return fig

def main():
    parser = argparse.ArgumentParser(description="Markov chain value model for StatsBomb open data")
    parser.add_argument("--competition", type=int, default=43, help="Competition ID (default: FIFA World Cup 2018)")
    parser.add_argument("--season", type=int, default=3, help="Season ID for the chosen competition")
    parser.add_argument("--max-matches", type=int, default=15, help="Number of matches to include (limits data size)")
    parser.add_argument("--output", type=Path, default=Path("value_model.html"), help="Interactive Plotly heatmap output")
    parser.add_argument("--static-output", type=Path, default=Path("value_model.png"), help="Static Matplotlib summary output")
    parser.add_argument("--support-output", type=Path, default=Path("support_value.html"), help="Support-aware interactive plot output")
    parser.add_argument("--progression-output", type=Path, default=Path("ball_progression.png"), help="Average movement quiver plot output")
    parser.add_argument("--shot-output", type=Path, default=Path("shot_quality.html"), help="Shot quality heatmap output")
    args = parser.parse_args()

    matches = load_matches(args.competition, args.season)
    if args.max_matches:
        matches = matches[: args.max_matches]

    print(f"Downloading events for {len(matches)} matches...")
    matches_events = []
    matches_with_frames: List[Tuple[List[dict], Dict[str, List[dict]]]] = []
    for match in matches:
        match_id = match["match_id"]
        events = load_match_events(match_id)
        frames = load_three_sixty(match_id)
        matches_events.append(events)
        matches_with_frames.append((events, frames))
        print(
            f"Loaded match {match_id} with {len(events)} events"
            + (f" and {sum(len(f) for f in frames.values())} freeze frames" if frames else "")
        )

    zone_counts, action_counts = build_transition_counts(matches_events)
    P, goal_probs, _ = counts_to_transition_matrix(zone_counts)

    state_values = solve_state_value(P, goal_probs, gamma=1.0)
    discounted_state_values = solve_state_value(P, goal_probs, gamma=0.9)
    action_values = compute_action_values(action_counts, state_values)

    fig = build_interactive_figure(state_values, discounted_state_values, action_values)
    fig.write_html(str(args.output), include_plotlyjs="cdn")
    print(f"Saved interactive visualisation to {args.output.resolve()}")

    if plt is not None:
        create_static_value_figure(state_values, discounted_state_values, action_values, args.static_output)
        print(f"Saved static summary to {args.static_output.resolve()}")
    else:
        print("Matplotlib not installed – skipping static summary output.")

    support_maps = compute_support_state_values(matches_with_frames)
    if support_maps:
        support_fig = build_support_figure(support_maps)
        support_fig.write_html(str(args.support_output), include_plotlyjs="cdn")
        print(f"Saved support-context visualisation to {args.support_output.resolve()}")
    else:
        print("No StatsBomb 360 freeze frames available for support-context visualisation.")

    vectors = compute_ball_progression_vectors(matches_events)
    if plt is not None and vectors:
        create_progression_plot(vectors, args.progression_output)
        print(f"Saved ball progression plot to {args.progression_output.resolve()}")
    elif plt is None:
        print("Matplotlib not installed – skipping ball progression plot.")
    else:
        print("Insufficient passing/carry data for progression plot.")

    dist_bins, angle_bins, probabilities = compute_shot_quality(matches_events)
    if probabilities.size:
        shot_fig = build_shot_quality_figure(dist_bins, angle_bins, probabilities)
        shot_fig.write_html(str(args.shot_output), include_plotlyjs="cdn")
        print(f"Saved shot quality visualisation to {args.shot_output.resolve()}")
    else:
        print("No shot data available for shot quality visualisation.")


if __name__ == "__main__":
    main()
