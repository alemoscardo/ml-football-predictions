"""Tune, select and evaluate the pre-match outcome models.

Seasons are split in time:
- the first season is burn-in only (Elo and form need history to warm up),
- the following seasons are the training set,
- the second-to-last season is the validation set used to pick the model,
- the last season is the untouched test set, scored once at the end.

Writes ``models/model_metrics.json``: the chosen model spec, every validation
trial, and the test-season scores next to the bookmaker's.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import MODEL_FEATURES, build_features

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
METRICS_PATH = MODELS_DIR / "model_metrics.json"
LABELS = ["A", "D", "H"]  # sorted, the column order log_loss expects
BOOKIE_COLUMNS = ["NormProbAway_B365", "NormProbDraw_B365", "NormProbHome_B365"]

SEARCH_SPACE = {
    "Logistic Regression": {"C": [0.01, 0.05, 0.2, 1.0]},
    "Extra Trees": {"max_depth": [4, 6, 8], "min_samples_leaf": [20, 50]},
}


def load_raw_matches() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files in data/ — run `python fetch_data.py` first.")
    frames = []
    for file in files:
        frame = pd.read_csv(file, encoding_errors="replace").copy()  # copy() defragments wide files
        frame["SeasonFile"] = file.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_dataset() -> pd.DataFrame:
    matches = build_features(load_raw_matches())
    return matches[matches["Result"].notna()].reset_index(drop=True)


def split_seasons(matches: pd.DataFrame) -> dict[str, list[str]]:
    seasons = sorted(matches["SeasonFile"].unique())
    if len(seasons) < 4:
        raise ValueError("Need at least four seasons: burn-in, train, validation, test.")
    return {
        "burn_in": seasons[:1],
        "train": seasons[1:-2],
        "validation": seasons[-2:-1],
        "test": seasons[-1:],
    }


def make_model(algorithm: str, params: dict[str, float]) -> Pipeline:
    if algorithm == "Logistic Regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, **params)),
            ]
        )
    if algorithm == "Extra Trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=400, max_features="sqrt", random_state=42, n_jobs=-1, **params
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def score(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    picks = np.array(LABELS)[proba.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(y_true, picks)),
        "log_loss": float(log_loss(y_true, proba, labels=LABELS)),
    }


def predict_sorted(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilities in LABELS order, whatever order the classifier stored."""
    proba = pd.DataFrame(model.predict_proba(X), columns=list(model.classes_))
    return proba[LABELS].to_numpy()


def tune(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> list[dict]:
    """Fit every candidate on train, score on validation; best log-loss first."""
    trials = []
    for algorithm, grid in SEARCH_SPACE.items():
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            model = make_model(algorithm, params).fit(train[features], train["Result"])
            trials.append(
                {
                    "algorithm": algorithm,
                    "params": params,
                    **score(validation["Result"], predict_sorted(model, validation[features])),
                }
            )
    return sorted(trials, key=lambda t: t["log_loss"])


def main() -> None:
    matches = load_dataset()
    split = split_seasons(matches)
    parts = {name: matches[matches["SeasonFile"].isin(seasons)] for name, seasons in split.items()}
    train, validation, test = parts["train"], parts["validation"], parts["test"]
    final_train = pd.concat([train, validation])

    report: dict[str, object] = {
        "seasons": split,
        "rows": {name: int(len(frame)) for name, frame in parts.items()},
        "bookmaker": {
            "validation": score(validation["Result"], validation[BOOKIE_COLUMNS].to_numpy()),
            "test": score(test["Result"], test[BOOKIE_COLUMNS].to_numpy()),
        },
    }

    trials = tune(train, validation, MODEL_FEATURES)
    best = trials[0]
    # Refit the chosen spec on train + validation, then score the test season once.
    model = make_model(best["algorithm"], best["params"]).fit(
        final_train[MODEL_FEATURES], final_train["Result"]
    )
    report["model"] = {
        "features": MODEL_FEATURES,
        "algorithm": best["algorithm"],
        "params": best["params"],
        "validation": {k: best[k] for k in ("accuracy", "log_loss")},
        "test": score(test["Result"], predict_sorted(model, test[MODEL_FEATURES])),
        "trials": trials,
    }

    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    bookie = report["bookmaker"]["test"]
    print(f"Train {', '.join(split['train'])} | validate {split['validation'][0]} | test {split['test'][0]}")
    print(f"  {'Bookmaker favourite':34s} acc {bookie['accuracy']:.3f}  log-loss {bookie['log_loss']:.4f}")
    entry = report["model"]
    label = f"Model ({entry['algorithm']})"
    print(f"  {label:34s} acc {entry['test']['accuracy']:.3f}  log-loss {entry['test']['log_loss']:.4f}")
    print(f"Saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
