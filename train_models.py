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


def split_temporal_holdout(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    seasons = sorted(matches["SeasonFile"].dropna().unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least two season files to run a temporal holdout.")

    test_season = seasons[-1]
    train_seasons = seasons[:-1]

    train_df = matches[matches["SeasonFile"].isin(train_seasons)].copy()
    test_df = matches[matches["SeasonFile"] == test_season].copy()

    train_df = train_df[train_df["Result"].notna()].copy()
    test_df = test_df[test_df["Result"].notna()].copy()

    return train_df, test_df, train_seasons, test_season


def evaluate_and_save_showcase_models(matches: pd.DataFrame) -> dict[str, object]:
    train_df, test_df, train_seasons, test_season = split_temporal_holdout(matches)
    _, tuned_logistic, tuned_extra_trees = build_models()

    model_runs = [
        ("Logistic Regression (Tuned)", "logistic_regression_showcase_model.pkl", tuned_logistic),
        ("Extra Trees (Tuned)", "extra_trees_showcase_model.pkl", tuned_extra_trees),
    ]

    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["Result"]
    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["Result"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_metrics: dict[str, dict[str, float | str]] = {}
    best_model = ""
    best_accuracy = -1.0

    for model_name, artifact_name, model in model_runs:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, preds))

        joblib.dump(model, MODELS_DIR / artifact_name)
        model_metrics[model_name] = {
            "artifact": artifact_name,
            "accuracy": accuracy,
        }

        if accuracy > best_accuracy:
            best_model = model_name
            best_accuracy = accuracy

    return {
        "train_seasons": ", ".join(train_seasons),
        "test_season": test_season,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": model_metrics,
        "current_model": best_model,
        "current_accuracy": best_accuracy,
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

    metrics = evaluate_and_save_showcase_models(prepared_matches)
    train_and_save_final_models(prepared_matches)

    metrics_path = MODELS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Temporal holdout evaluation")
    print(f"Train seasons: {metrics['train_seasons']}")
    print(f"Test season : {metrics['test_season']}")
    print("Model accuracies:")
    for model_name, model_info in metrics["models"].items():
        print(f"  - {model_name}: {model_info['accuracy']:.4f} ({model_info['artifact']})")
    print(f"Current model: {metrics['current_model']}")
    print(f"Current accuracy: {metrics['current_accuracy']:.4f}")
    print(f"Saved model artifacts to: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
