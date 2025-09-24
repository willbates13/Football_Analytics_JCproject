# Football Analytics Markov Model Demo

This repository contains a single-script demo that downloads StatsBomb's open
data and builds a Markov chain model inspired by Ian Graham's possession value
work. The output is an interactive Plotly heatmap that lets you explore how the
value of different on-ball actions changes across the pitch.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python markov_value_model.py --max-matches 20 --output value_model.html
```

The script saves an HTML file (`value_model.html` by default) containing the
interactive visualisation, which you can embed directly into a presentation.

### Command line options

* `--competition` – StatsBomb competition ID (default: `43`, 2018 World Cup).
* `--season` – Season ID for the chosen competition (default: `3`).
* `--max-matches` – Limit the number of matches downloaded to keep the example
  lightweight.
* `--output` – Path to write the interactive Plotly HTML file.

The dropdown menu in the resulting figure includes:

* **State Value** – probability of scoring before the possession ends from each
  zone (the Ian Graham-style possession value).
* **Discounted Value** – an RL-inspired variant with a discount factor that
  emphasises quicker scoring opportunities.
* **Action Value** heatmaps – expected goal value if a possession chooses a
  specific action (pass, carry, shot, etc.) from each zone.
* **Action Advantage** heatmaps – the improvement over the average action in
  that zone, analogous to an RL advantage function.
