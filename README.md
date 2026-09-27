[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alemoscardo-ml-football-predictions-streamlit-app-vkbxgd.streamlit.app/)

# Football Match Outcome Predictor

Machine learning app to predict English Premier League outcomes (`H`, `D`, `A`) from match statistics.

![Dashboard](docs/dashboard.png)

## Overview

This repo contains:
- A reproducible training pipeline (`train_models.py`)
- Shared feature engineering (`feature_engineering.py`)
- A Streamlit app (`streamlit_app.py`) that showcases trained models on a fixed temporal holdout dataset

## The App

One page, built for a quick read:
- **Headline KPIs:** accuracy and log-loss on the unseen season, each against the bookmaker
- **How it compares:** the model next to random guessing, always backing the home side and the
  bookmaker favourite
- **How it behaves:** running accuracy across the season, confidence vs. hit rate, a confusion
  matrix and permutation feature importance
- **Every match:** filterable table (team, misses, matches where the model beat the bookmaker);
  select a row to compare model and bookmaker probabilities, or export to CSV

Models are trained in-process at startup (about a second), so the app never depends on the
scikit-learn version that wrote a pickle.

## What Changed

The modeling pipeline was upgraded to improve prediction quality:
- Added consistent feature engineering for both training and inference
- Added bookmaker-odds features (`B365H`, `B365D`, `B365A`) and derived implied-probability features
- Switched to a tuned `ExtraTreesClassifier`
- Added temporal holdout evaluation to better reflect real forecasting

## Showcase Evaluation

Temporal holdout setup:
- Train: `E0_2122`, `E0_2223`
- Test: `E0_2324`

| Model | Accuracy |
|---|---:|
| Logistic Regression (Tuned) | 61.05% |
| Extra Trees (Tuned) | **65.00%** |

## Data Source

- [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)

## Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run streamlit_app.py
```
