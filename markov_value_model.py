"""Simple Markov chain value model for StatsBomb open data.

This script downloads a sample of StatsBomb open-data matches, builds a
possession-based Markov chain that approximates Ian Graham's possession value
model, and produces an interactive Plotly heatmap that visualises the value of
different actions from different pitch locations.

Usage (from the repository root):

    python markov_value_model.py --max-matches 20 --output value_model.html

The script keeps the file structure intentionally simple so it can be dropped
into presentations or notebooks without additional project scaffolding.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import requests

STATS_BOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Pitch and grid settings (StatsBomb pitch: 120 x 80 coordinates)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_X = 6
GRID_Y = 4


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


def build_interactive_figure(
    state_values: np.ndarray,
    discounted_state_values: np.ndarray,
    action_values: Dict[Tuple[int, str], float],
) -> go.Figure:
    """Create a Plotly figure with interactive dropdowns for different metrics."""

    num_zones = GRID_X * GRID_Y
    zone_value_map = {zone: state_values[zone] for zone in range(num_zones)}
    discounted_map = {zone: discounted_state_values[zone] for zone in range(num_zones)}

    heatmaps = []
    labels = []

    heatmaps.append(grid_from_values(zone_value_map))
    labels.append("State Value (xG probability)")

    heatmaps.append(grid_from_values(discounted_map))
    labels.append("Discounted Value (quick chance focus)")

    # Aggregate action values and advantage (RL-style improvement over average)
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
            heatmaps.append(grid_from_values(action_map))
            labels.append(f"Action Value: {action}")

        if advantage_map:
            heatmaps.append(grid_from_values(advantage_map))
            labels.append(f"Action Advantage: {action}")

    x_labels = [f"Zone {x + 1}" for x in range(GRID_X)]
    y_labels = [f"Band {GRID_Y - y}" for y in range(GRID_Y)]

    traces = []
    for heatmap in heatmaps:
        traces.append(
            go.Heatmap(
                z=heatmap,
                x=x_labels,
                y=y_labels,
                colorscale="YlOrRd",
                zmin=0,
                zmax=float(np.nanmax(heatmap)) if np.isfinite(np.nanmax(heatmap)) else 1.0,
                colorbar=dict(title="xG"),
                visible=False,
            )
        )

    if traces:
        traces[0].visible = True

    fig = go.Figure(data=traces)

    buttons = []
    for idx, label in enumerate(labels):
        visibility = [False] * len(traces)
        visibility[idx] = True
        buttons.append(dict(label=label, method="update", args=[{"visible": visibility}, {"title": label}]))

    fig.update_layout(
        title=labels[0] if labels else "Markov Chain Value Model",
        xaxis=dict(title="Pitch length (left to right)", side="top"),
        yaxis=dict(title="Pitch width", autorange="reversed"),
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True)],
        template="plotly_white",
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Markov chain value model for StatsBomb open data")
    parser.add_argument("--competition", type=int, default=43, help="Competition ID (default: FIFA World Cup 2018)")
    parser.add_argument("--season", type=int, default=3, help="Season ID for the chosen competition")
    parser.add_argument("--max-matches", type=int, default=15, help="Number of matches to include (limits data size)")
    parser.add_argument("--output", type=Path, default=Path("value_model.html"), help="Where to save the interactive plot")
    args = parser.parse_args()

    matches = load_matches(args.competition, args.season)
    if args.max_matches:
        matches = matches[: args.max_matches]

    print(f"Downloading events for {len(matches)} matches...")
    matches_events = []
    for match in matches:
        match_id = match["match_id"]
        events = load_match_events(match_id)
        matches_events.append(events)
        print(f"Loaded match {match_id} with {len(events)} events")

    zone_counts, action_counts = build_transition_counts(matches_events)
    P, goal_probs, _ = counts_to_transition_matrix(zone_counts)

    state_values = solve_state_value(P, goal_probs, gamma=1.0)
    discounted_state_values = solve_state_value(P, goal_probs, gamma=0.9)
    action_values = compute_action_values(action_counts, state_values)

    fig = build_interactive_figure(state_values, discounted_state_values, action_values)
    fig.write_html(str(args.output), include_plotlyjs="cdn")

    print(f"Saved interactive visualisation to {args.output.resolve()}")


if __name__ == "__main__":
    main()
