[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alemoscardo-ml-football-predictions-streamlit-app-vkbxgd.streamlit.app/)
[![tests](https://github.com/alemoscardo/ml-football-predictions/actions/workflows/tests.yml/badge.svg)](https://github.com/alemoscardo/ml-football-predictions/actions/workflows/tests.yml)

# Premier League Outcome Model

Forecasts English Premier League results (home win / draw / away win) **before kick-off**,
using only information available at the time, and measures the forecasts against the
bookmaker on a season the model never saw.

![Dashboard](docs/dashboard.png)

## Results

Test season **2025/26** (380 matches), scored once after model selection on 2024/25:

| Approach | Accuracy | Log-loss |
|---|---:|---:|
| Random guess | 33.3% | 1.099 |
| Always back the home side | 42.6% | — |
| **Bookmaker favourite (Bet365)** | **48.9%** | **1.019** |
| Model: Form & Elo (logistic regression) | 48.2% | 1.027 |
| Model: Form, Elo & odds (logistic regression) | 47.6% | 1.026 |

Built only from public match history, the model gets within 0.01 log-loss of the bookmaker,
whose odds also price in line-ups, injuries and market money. Adding the bookmaker's own
probabilities as features barely moves the score: the market already knows what the form
features know.

## How it works

**Features** (`feature_engineering.py`), each computed only from matches played before the one
being predicted:
- **Elo ratings** updated after every result (goal-margin weighted, home advantage, 20% pull
  towards the mean between seasons, clubs new to the data start below average)
- **Recent form**: points, goals scored and conceded, shots on target for and against over each
  side's last 5 league matches, plus the home-minus-away gaps
- **Rest days** since each side's previous match
- *Optional*: Bet365 odds converted to margin-free implied probabilities

**Validation** (`train_models.py`) is strictly chronological:

| Seasons | Role |
|---|---|
| 2014/15 | Burn-in: warms up Elo and form, never trained on |
| 2015/16 – 2023/24 | Training |
| 2024/25 | Validation: picks algorithm and hyper-parameters by log-loss |
| 2025/26 | Test: scored once at the end |

Candidates are logistic regression (several regularisation strengths) and extra trees (depth and
leaf-size grid). The winner is refitted on train + validation before the test season is scored.
Every trial is saved to `models/model_metrics.json` and shown in the app.

## The app

- **Headline KPIs**: accuracy and log-loss, each against the bookmaker
- **How it compares**: the model next to random guessing, always backing the home side and the
  bookmaker favourite
- **How it behaves**: running accuracy across the season, confidence vs. hit rate (calibration),
  confusion matrix and permutation feature importance
- **Every match**: filterable table; select a row to compare the model's and the bookmaker's
  probabilities, or export to CSV
- **Method**: validation design, limitations and the full model-selection table

Models are refitted in-process at startup from the saved spec, so the app never depends on the
scikit-learn version that wrote a pickle.

## Notebook

[`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb): outcome shares per
season, Elo sanity check, a leakage test that recomputes rolling form by hand, and a calibration
plot of model vs. bookmaker.

## Tests

`tests/test_features.py` guards against look-ahead leakage: it tampers with one match's result
and checks that neither that match's features nor any earlier ones change, and recomputes rolling
form by hand. GitHub Actions runs the tests and the full training pipeline on every push.

## Limitations

- No line-ups, injuries or expected-goals data, which is where bookmakers get their edge
- Draws are almost never the single most likely outcome, so they are rarely predicted
- One test season is 380 matches: differences of a point or two in accuracy are within noise

## Run locally

```bash
pip install -r requirements.txt
python fetch_data.py        # download season CSVs from Football-Data.co.uk into data/
python train_models.py      # tune on 2024/25, score 2025/26, write models/model_metrics.json
streamlit run streamlit_app.py

pip install -r requirements-dev.txt && pytest   # tests
```

## Data

[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php): results, match statistics
and pre-match odds for every Premier League match.
