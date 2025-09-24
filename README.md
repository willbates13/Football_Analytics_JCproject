# Football Analytics Markov Model Demo

This repository contains a single-script demo that downloads StatsBomb's open
data and builds a Markov chain model inspired by Ian Graham's possession value
work. The refreshed version adds higher-resolution pitch grids, goal markers,
support-context modelling from StatsBomb 360 freeze frames and additional
visualisations for presentations (interactive and static).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python markov_value_model.py --max-matches 20
```

The script saves multiple outputs by default:

* `value_model.html` – interactive Plotly heatmap of state/action values.
* `value_model.png` – static Matplotlib summary of key value surfaces.
* `support_value.html` – contextual model that folds StatsBomb 360 player
  positions into the state (falls back gracefully if freeze frames are
  unavailable).
* `ball_progression.png` – quiver plot describing average pass/carry movement.
* `shot_quality.html` – heatmap of goal probability by shot distance/angle.

### Command line options

* `--competition` – StatsBomb competition ID (default: `43`, 2018 World Cup).
* `--season` – Season ID for the chosen competition (default: `3`).
* `--max-matches` – Limit the number of matches downloaded to keep the example
  lightweight.
* `--output` – Path to write the interactive Plotly HTML file.
* `--static-output` – File path for the static Matplotlib summary.
* `--support-output` – Where to save the support-context HTML visualisation.
* `--progression-output` – Output path for the quiver plot.
* `--shot-output` – File path for the shot quality heatmap.

The dropdown menu in the interactive figure includes:

* **State Value** – probability of scoring before the possession ends from each
  zone (the Ian Graham-style possession value).
* **Discounted Value** – an RL-inspired variant with a discount factor that
  emphasises quicker scoring opportunities.
* **Action Value** heatmaps – expected goal value if a possession chooses a
  specific action (pass, carry, shot, etc.) from each zone.
* **Action Advantage** heatmaps – the improvement over the average action in
  that zone, analogous to an RL advantage function.

The support-context visualisation groups states by the on-ball player's help:
isolated versus supported, pressure levels and whether a progressive passing
lane exists. These states are derived from StatsBomb 360 freeze frames, so the
plot only appears for competitions where those data are available.

The static PNG and quiver plot add presentation-friendly slides for exploring
action values and ball movement without relying on interactive controls. The
shot quality heatmap complements the possession model by answering "what kinds
of shots score most often?" in terms of distance and shooting angle.
