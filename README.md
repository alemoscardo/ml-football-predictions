[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://alemoscardo-ml-football-predictions-streamlit-app-vkbxgd.streamlit.app/)

# Football Match Outcome Predictor

Machine learning app to predict English Premier League outcomes (`H`, `D`, `A`) from match statistics.

## Overview

This repo contains:
- A reproducible training pipeline (`train_models.py`)
- Shared feature engineering (`feature_engineering.py`)
- A Streamlit app (`streamlit_app.py`) for CSV and manual predictions

## What Changed

The modeling pipeline was upgraded to improve prediction quality:
- Added consistent feature engineering for both training and inference
- Added bookmaker-odds features (`B365H`, `B365D`, `B365A`) and derived implied-probability features
- Switched to a tuned `ExtraTreesClassifier`
- Added temporal holdout evaluation to better reflect real forecasting

## Model Performance

Temporal holdout setup:
- Train: `E0_2122`, `E0_2223`
- Test: `E0_2324`

| Current Best Model | Accuracy |
|---|---:|
| Tuned Extra Trees | **65.00%** |

## Data Source

- [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)

## Run Locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run streamlit_app.py
```
