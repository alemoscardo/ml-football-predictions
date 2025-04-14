
# Football Match Outcome Predictor

**A machine learning application that predicts the result of English Premier League matches using match statistics.**

---

## Overview

This project is designed to simulate a real-world sports analytics pipeline. It focuses on predicting match outcomes — Home Win (H), Draw (D), or Away Win (A) — using structured match data such as:

- Shots (total and on target)
- Corners, fouls, yellow/red cards
- Goal difference and derived features

It includes a multi-model comparison, interactive user interface, and CSV input support — ideal for demonstrating applied machine learning and user-facing deployment.

---

## Key Features

| Feature | Description |
|--------|-------------|
| **Multi-class Classification** | Predict outcome: Home Win / Draw / Away Win |
| **Model Selection** | Logistic Regression (default) or Random Forest |
| **CSV Upload** | Upload real match data for batch predictions |
| **Manual Input** | Simulate individual matches via form entry |
| **Evaluation Tools** | Accuracy summary, prediction charts, confusion matrix |
---

## Model Performance

Trained using historical EPL data with engineered features.

| Model                 | Accuracy |
|----------------------|----------|
| Logistic Regression  | ~61%     |
| Random Forest        | ~56%     |

---


## Data Source

- [Football-Data.co.uk](https://www.football-data.co.uk/englandm.php)  
  Historical English Premier League stats (public datasets)

> ✅ This project was built as part of my portfolio to demonstrate practical machine learning and data product development skills.
