# project_eleven — Phase 2: LaLiga match outcome prediction

Predicting the outcome of Spanish first division matches (home win / draw /
away win) using machine learning, based on real match data.

## Goal

Using the history of LaLiga matches (2020-2021 to 2025-2026 seasons), build
features that are available before kickoff (Elo rating, recent form, xG,
rest days), then train and evaluate a three-class classification model.
## Data

Matches come from [fbref.com](https://fbref.com). The repo
versions, in `data/`, one CSV schedule file per season, one row per match
played, with the original columns (date, teams, home/away xG, score, etc.).

## Project structure

```
project_eleven/
├── data/                                
│   ├── laliga_2020_2021_fixtures.csv      # one schedule CSV per season
│   ├── laliga_2021_2022_fixtures.csv      
│   ├── laliga_2022_2023_fixtures.csv
│   ├── laliga_2023_2024_fixtures.csv
│   ├── laliga_2024_2025_fixtures.csv
│   └── laliga_2025_2026_fixtures.csv
├── src/                                    
│   ├── __init__.py
│   └── features.py                        # build_pre_match_features()
├── notebook_project_eleven_phase_two.ipynb # full pipeline + results
├── requirements.txt                        # Pinned dependencies
├── .gitignore
└── README.md
```

## Installation and usage

```bash
# 1. Clone the repo and cd into it
git clone <repo-url> && cd project_eleven

# 2. Virtual environment + dependencies
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt

# 3. Run the notebook
jupyter lab notebook_project_eleven_phase_two.ipynb
```

The notebook is designed to run in one go: Kernel → Restart & Run All
regenerates all intermediate files and results from the CSVs in `data/`
alone.

## Methodology

### Features built (pre-match)

| Feature | Description |
|---|---|
| `elo_home`, `elo_away` | Elo rating of each team, updated match after match |
| `elo_diff` | Elo difference, home minus away |
| `form_points_home_5`, `form_points_away_5` | Average points over the last 5 matches (win 3, draw 1.5, loss 0) |
| `form_goal_diff_home_5`, `form_goal_diff_away_5` | Average goal difference over the last 5 matches |
| `form_xG_home_5`, `form_xG_away_5` | Average xG over the last 5 matches |
| `xG_diff` | Difference between average xG, home minus away |
| `rest_diff` | Difference in rest days between the two teams |
| `day` | Day of the week of the match (integer encoded) |

All of these features are computed before the Elo and match history are
updated with the current match: a match never uses information coming from
itself or from the future (no temporal leakage). This is checked explicitly
by an anti-leakage test (see below).

### Evaluation (single split)

- Temporal split: training on the first 3 seasons (2020-2021 to 2022-2023,
  1140 matches), testing on the next 2 (2023-2024 and 2024-2025, 760
  matches). The 2025-2026 season (partial, no xG) is left out.
- Model: multinomial logistic regression, inside a `Pipeline` (median
  imputation → normalization → classifier) so that the preprocessing is
  learned only on the training set.
- Baselines: (1) class prior, always predict the most frequent outcome
  ("home win"); (2) Elo only, same model using only the `elo_diff` feature.
- Metrics: accuracy, log loss, confusion matrix.

The Elo `K` factor is tuned by a sweep (accuracy / log loss curves on the
same test set), which confirms `K = 20`.

### Walk-forward validation

A single train/test split is one measurement point, sensitive to which
seasons end up in the test set. To get a more robust estimate, the model is
also evaluated with a walk-forward scheme that replays "train on the past,
test on the next season" across all seasons:

- expanding window: train on every season before the test season;
- sliding window: train on only the 2 previous seasons, to see whether
  older data helps or hurts.

Each fold is checked against the same two baselines (prior, Elo only), the
per-fold scores are aggregated as mean plus or minus standard deviation, and
a bootstrap confidence interval is computed on the pooled out-of-fold
predictions. An anti-leakage test rebuilds the features on a prefix of the
data and checks, match by match, that they match the ones computed on the
full dataset, confirming that no feature depends on the future.

### Handling the draw problem

The single-split evaluation showed that the model almost never predicts a
draw. Three fixes are tested on the same expanding walk-forward scheme,
each judged with a metric suited to the question:

- RPS : unlike log loss, RPS takes into account
  the order loss < draw < win, and is the standard metric in football
  forecasting.
- `class_weight="balanced"`, to force the logistic regression to pay more
  attention to the minority class (the draw).
- A Dixon-Coles goals model: home and away goals are each modeled as
  Poisson variables driven by per-team attack/defense strengths fitted by
  maximum likelihood, with a low-score correction (parameter rho) that
  specifically boosts 0-0 and 1-1 draws. match outcome probabilities are
  then derived from the resulting score grid.

## Results

### Single split (2 test seasons, 760 matches)

| Model | Accuracy | Log loss |
|---|---:|---:|
| Logistic regression (all features) | 0.536 | 0.974 |
| Baseline, prior (always home) | 0.442 | 1.073 |
| Baseline, Elo only (`elo_diff`) | 0.536 | 0.967 |

Two takeaways, stated plainly:

1. The model clearly beats the prior (53.6% vs 44.2%), so it does learn a
   real signal. But Elo alone does just as well, and even slightly better
   on log loss. Form, xG, and rest features bring almost nothing on top of
   the level difference (the full model overfits a bit). The coefficient
   analysis confirms this: `elo_diff`, `elo_away`, and `elo_home` dominate.
2. The model almost never predicts a draw (recall close to 0 on that
   class). This is the classic problem with draws in football prediction.

### Walk-forward validation

The walk-forward average (accuracy around 0.53, log loss around 1.01 for
the expanding scheme) sits slightly below the single split's 0.536: that
split happened to test on the most recent seasons, which benefit from the
largest training history. Walk-forward gives a more representative estimate
together with its uncertainty (95% CI on accuracy roughly [0.50, 0.55]).
The first fold, with only one training season, is the weakest one, but the
benefit of extra history plateaus quickly beyond 2 to 3 seasons. Elo only
keeps its edge on every single fold, with a log loss systematically at or
below the full model's, confirming that the single-split finding was not an
artifact of one particular cut: form, xG, and rest add nothing on top of
the Elo difference. The sliding window (2 seasons) does at least as well as
the expanding one on log loss, suggesting that older seasons do not add
much, consistent with a sport that evolves from year to year.

### The draw problem

None of the three fixes beats Elo only, even measured by RPS:

| Model | Accuracy | Log loss | RPS | Draw recall | Avg P(draw) |
|---|---:|---:|---:|---:|---:|
| Full LR | 0.526 | 1.012 | 0.202 | 0.02 | 0.257 |
| Balanced LR | 0.513 | 1.036 | 0.207 | 0.24 | 0.305 |
| Elo only | 0.528 | 0.994 | 0.199 | 0.00 | 0.265 |
| Dixon-Coles | 0.480 | 1.050 | 0.218 | 0.02 | 0.271 |

- The draw problem is not a calibration problem. Every model assigns an
  average probability of about 0.26 to the draw, which is essentially the
  actual draw rate (0.266). Draws are simply almost never the single most
  likely outcome for a given match, which is why recall stays close to 0.
  This is an intrinsic property of football, not a flaw in the model.
- `class_weight="balanced"` raises draw recall (0.02 to 0.24) but by 
  over-predicting draws (average P(draw) rises to 0.305, above the actual 
  0.266), which degrades both log loss and RPS.
- Dixon-Coles underperforms here not because the model is wrong (the fit
  converges, rho correctly boosts low scores, home advantage is estimated
  at around 0.23), but because it discards the Elo signal, the most
  predictive one on this data, in favor of goal strengths estimated from
  only a few seasons.

RPS confirms the log loss ranking (Elo ahead), and the draw resists every
direct approach tried so far. The genuinely promising direction is not a
better standalone draw model, but a hybrid one: adding the Dixon-Coles
expected goals or its P(draw) as extra features next to `elo_diff`, to
combine Elo's discriminative power with the Dixon-Coles goals structure.

## Known limitations

- Cold start. For a team's first 5 matches, the form features are
  unavailable and return 0 (average over an empty history). These early
  rows are therefore degraded. A possible improvement would be to exclude
  them or initialize form with a prior.
- Elo with no reset between seasons. Elo runs continuously across all
  seasons. Promoted teams inherit the base Elo (1500) with no regression
  to the mean between seasons.
- 2025-2026 season excluded from evaluation: partial, and no xG column on
  fbref (its xG would have to be imputed, which would hurt the signal).
- The Elo signal dominates every other feature and every fix tried for the
  draw problem, which points to a ceiling in the current feature set rather
  than a modeling issue.

## Roadmap

- Hybrid draw model: feed the Dixon-Coles expected goals or its P(draw) as
  extra features alongside `elo_diff`, instead of choosing between an Elo
  based classifier and a goals based one.
- Richer features: since form, xG, and rest add little beyond Elo, look for
  features that carry information Elo does not already capture (for
  example lineup or injury data, head to head history).
- Probability calibration: Brier score, reliability curve, now that a
  walk-forward estimate with confidence intervals is available to calibrate
  against.
- Comparison with bookmaker odds.
