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
* trains a lightweight Q-learning policy in parallel that suggests pass / shoot /
  dribble actions for the ball carrier using the same freeze-frame context
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
* `--q-gamma` / `--q-alpha` – control the discount factor and learning rate for
  the Q-learning policy updates
* `--epsilon`, `--epsilon-decay`, `--epsilon-min` – configure epsilon-greedy
  exploration for the suggested actions
* `--pretrain-count` – automatically warm up the Q-table on this many additional
  matches before rendering the target replay (skipping the target match)
* `--pretrain-match-id` – specify match IDs manually for pretraining (can be
  repeated)
* `--rl-seed` – fix the pseudo-random exploration sequence for reproducibility
* `--output` – choose a different HTML file name

Because the animation is driven entirely by StatsBomb 360 freeze frames, some
competitions will not appear unless they have that extra contextual data. Use
`--list` to confirm availability before rendering.

## How the Q-learning demo works

Every freeze-framed event is treated as a reinforcement learning state. The
state vector is fully derived from the freeze frame: it records the ball
carrier's grid cell, the number of team-mates and opponents in each grid cell
and a discretised estimate of the distance to the nearest defender. Actions are
restricted to the three intuitive decisions visible in the data – pass, shoot
or dribble (carry).

Rewards stay simple for interpretability: goals are worth `1.0`, shots are
weighted by their StatsBomb xG, assists receive a positive bonus and losing
possession incurs a penalty. A tabular Q-learning agent updates on top of the
observed sequence using:

```
Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
```

where `α`/`γ` are controlled by `--q-alpha` and `--q-gamma`. Suggested actions in
the animation come from an epsilon-greedy policy (configurable via
`--epsilon`, `--epsilon-decay` and `--epsilon-min`). If you want stronger
recommendations, warm up the Q-table across a few matches with
`--pretrain-count` or `--pretrain-match-id` so the agent has seen more examples
before the target replay. The overlay shows both the agent's recommendation and
the action that actually happened so you can compare policy learning against
the real match footage.
