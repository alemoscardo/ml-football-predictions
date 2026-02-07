from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import MODEL_FEATURES, prepare_matches_dataframe

DATA_GLOB = "E0_*.csv"
MODELS_DIR = Path("models")


def load_raw_matches() -> pd.DataFrame:
    files = sorted(Path("data").glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError("No CSV files found in data/ with pattern E0_*.csv")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        frame["SeasonFile"] = file.stem
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    legacy_logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    tuned_logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )

    tuned_extra_trees = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    criterion="entropy",
                    random_state=42,
                ),
            ),
        ]
    )

    return legacy_logistic, tuned_logistic, tuned_extra_trees


def evaluate_temporal_holdout(matches: pd.DataFrame) -> dict[str, float | str]:
    seasons = sorted(matches["SeasonFile"].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least two season files to run a temporal holdout.")

    test_season = seasons[-1]
    train_seasons = seasons[:-1]

    train_df = matches[matches["SeasonFile"].isin(train_seasons)].copy()
    test_df = matches[matches["SeasonFile"] == test_season].copy()

    train_df = train_df[train_df["Result"].notna()].copy()
    test_df = test_df[test_df["Result"].notna()].copy()

    _, _, tuned_extra_trees = build_models()

    tuned_extra_trees.fit(train_df[MODEL_FEATURES], train_df["Result"])
    tuned_extra_trees_pred = tuned_extra_trees.predict(test_df[MODEL_FEATURES])
    tuned_extra_trees_accuracy = accuracy_score(test_df["Result"], tuned_extra_trees_pred)

    return {
        "train_seasons": ", ".join(train_seasons),
        "test_season": test_season,
        "current_model": "Tuned Extra Trees",
        "current_accuracy": tuned_extra_trees_accuracy,
    }


def train_and_save_final_models(matches: pd.DataFrame) -> None:
    _, tuned_logistic, tuned_extra_trees = build_models()

    full_df = matches[matches["Result"].notna()].copy()
    X_full = full_df[MODEL_FEATURES]
    y_full = full_df["Result"]

    tuned_logistic.fit(X_full, y_full)
    tuned_extra_trees.fit(X_full, y_full)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(MODEL_FEATURES, MODELS_DIR / "features_used.pkl")
    joblib.dump(tuned_logistic, MODELS_DIR / "logistic_regression_model.pkl")
    joblib.dump(tuned_extra_trees, MODELS_DIR / "extra_trees_model.pkl")
    # Backward-compatible alias for older app versions.
    joblib.dump(tuned_extra_trees, MODELS_DIR / "random_forest_model.pkl")


def main() -> None:
    raw_matches = load_raw_matches()
    prepared_matches = prepare_matches_dataframe(raw_matches)

    metrics = evaluate_temporal_holdout(prepared_matches)
    train_and_save_final_models(prepared_matches)

    metrics_path = MODELS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Temporal holdout evaluation")
    print(f"Train seasons: {metrics['train_seasons']}")
    print(f"Test season : {metrics['test_season']}")
    print(f"Current model: {metrics['current_model']}")
    print(f"Current accuracy: {metrics['current_accuracy']:.4f}")
    print(f"Saved model artifacts to: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
