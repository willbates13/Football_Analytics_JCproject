# StatsBomb 360 Markov Chain Visualiser

This repository keeps everything in a single Python script that pulls directly
from [StatsBomb's open data](https://github.com/statsbomb/open-data) and builds a
transparent Markov decision process (MDP) style value model. Every state uses
StatsBomb 360 freeze-frame player locations so you can literally watch the value
surface emerge event by event.

## What the script does

* downloads competitions, matches and events until it finds ones with StatsBomb
  360 freeze frames (no other data sources are used)
* builds a pitch grid and initialises the value function at zero
* treats each freeze-framed event as a state whose features are the teammate and
  opponent locations plus the ball position
* updates the value grid sequentially with a temporal-difference rule so you can
  step through the creation of the Markov value model
* draws every player and the ball on a stylised pitch alongside the evolving
  value surface
* saves the result as a Plotly HTML animation where you can scrub through or
  autoplay the match

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python markov_value_model.py --max-matches 10
```

The script prints the path to the generated HTML file (default
`three_sixty_markov.html`). Open it in a browser to step through each event.

## Useful command-line options

* `--list` – show the matches (within the search limits) that include 360 data
  and exit
* `--match-id` – render a specific match that is known to have StatsBomb 360
  freeze frames
* `--competition-id` / `--season-id` – filter the search to a particular
  competition and season
* `--max-matches` – limit how many matches are checked for 360 data (default 5)
* `--grid-x` / `--grid-y` – control the pitch discretisation used for the value
  function
* `--gamma` / `--alpha` – tweak the temporal-difference discount factor and
  learning rate
* `--output` – choose a different HTML file name

Because the animation is driven entirely by StatsBomb 360 freeze frames, some
competitions will not appear unless they have that extra contextual data. Use
`--list` to confirm availability before rendering.
