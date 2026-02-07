from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from feature_engineering import MODEL_FEATURES, build_feature_matrix, prepare_matches_dataframe

st.set_page_config(page_title="Football Model Showcase", layout="wide")

LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

MODEL_OPTIONS = {
    "Extra Trees (Tuned)": {
        "showcase_artifact": MODELS_DIR / "extra_trees_showcase_model.pkl",
        "fallback_artifact": MODELS_DIR / "extra_trees_model.pkl",
    },
    "Logistic Regression (Tuned)": {
        "showcase_artifact": MODELS_DIR / "logistic_regression_showcase_model.pkl",
        "fallback_artifact": MODELS_DIR / "logistic_regression_model.pkl",
    },
}


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, object]:
    metrics_path = MODELS_DIR / "model_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_showcase_dataset(metrics: dict[str, object]) -> Path:
    test_season = metrics.get("test_season")
    if isinstance(test_season, str):
        candidate = DATA_DIR / f"{test_season}.csv"
        if candidate.exists():
            return candidate

    files = sorted(DATA_DIR.glob("E0_*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in data/ with pattern E0_*.csv")
    return files[-1]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_resource(show_spinner=False)
def load_features() -> list[str]:
    features_path = MODELS_DIR / "features_used.pkl"
    if features_path.exists():
        return joblib.load(features_path)
    return MODEL_FEATURES


st.title("Football Match Outcome Model Showcase")
st.caption("Predictions are evaluated on a temporal holdout season not used during showcase training.")

metrics = load_metrics()
features = load_features()

try:
    showcase_dataset = resolve_showcase_dataset(metrics)
    df_raw = pd.read_csv(showcase_dataset)
except Exception as exc:
    st.error(f"Unable to load showcase dataset: {exc}")
    st.stop()

model_choice = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
model_paths = MODEL_OPTIONS[model_choice]
showcase_model_path = model_paths["showcase_artifact"]
fallback_model_path = model_paths["fallback_artifact"]

if showcase_model_path.exists():
    model_path = showcase_model_path
    using_showcase_model = True
else:
    model_path = fallback_model_path
    using_showcase_model = False

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error(f"Unable to load model `{model_path}`: {exc}")
    st.stop()

if using_showcase_model:
    st.success(f"Loaded showcase artifact: `{model_path.name}`")
else:
    st.warning(
        f"Showcase artifact not found for {model_choice}. "
        f"Using fallback artifact `{model_path.name}`."
    )

info_col_1, info_col_2, info_col_3 = st.columns(3)
info_col_1.metric("Showcase Dataset", showcase_dataset.name)
info_col_2.metric("Train Seasons", str(metrics.get("train_seasons", "n/a")))
info_col_3.metric("Test Season", str(metrics.get("test_season", showcase_dataset.stem)))

df_processed = prepare_matches_dataframe(df_raw)
X = build_feature_matrix(df_raw, features)
preds = pd.Series(model.predict(X), index=df_processed.index)

confidence = None
if hasattr(model, "predict_proba"):
    confidence = pd.Series(model.predict_proba(X).max(axis=1), index=df_processed.index)

results = df_processed.copy()
results["Predicted"] = preds.map(LABEL_MAP).fillna(preds.astype(str))

if confidence is not None:
    results["Confidence"] = confidence.mul(100).round(1).map(lambda value: f"{value:.1f}%")

if "Result" in df_processed.columns:
    results["Actual"] = df_processed["Result"].map(LABEL_MAP)
    results["Correct"] = (preds == df_processed["Result"]).map({True: "Yes", False: "No"})

st.subheader("Showcase Results")

if "Result" in df_processed.columns:
    mask = df_processed["Result"].notna()
    if mask.any():
        accuracy = float((preds[mask] == df_processed.loc[mask, "Result"]).mean())

        score_col_1, score_col_2, score_col_3 = st.columns(3)
        score_col_1.metric("Matches Evaluated", int(mask.sum()))
        score_col_2.metric("Accuracy", f"{accuracy * 100:.2f}%")

        saved_accuracy = None
        model_metrics = metrics.get("models")
        if isinstance(model_metrics, dict):
            selected_metrics = model_metrics.get(model_choice)
            if isinstance(selected_metrics, dict):
                maybe_accuracy = selected_metrics.get("accuracy")
                if isinstance(maybe_accuracy, (float, int)):
                    saved_accuracy = float(maybe_accuracy)
        score_col_3.metric(
            "Saved Holdout Accuracy",
            "n/a" if saved_accuracy is None else f"{saved_accuracy * 100:.2f}%",
        )

        st.subheader("Prediction vs Actual Distribution")
        counts = pd.DataFrame(
            {
                "Predicted": preds[mask].map(LABEL_MAP).value_counts(),
                "Actual": df_processed.loc[mask, "Result"].map(LABEL_MAP).value_counts(),
            }
        ).fillna(0)
        st.bar_chart(counts)

        confusion_df = pd.crosstab(
            df_processed.loc[mask, "Result"].map(LABEL_MAP),
            preds[mask].map(LABEL_MAP),
            rownames=["Actual"],
            colnames=["Predicted"],
        )
        st.subheader("Confusion Matrix")
        st.dataframe(confusion_df)

display_columns = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c in results.columns]
display_columns.append("Predicted")
if "Confidence" in results.columns:
    display_columns.append("Confidence")
if "Actual" in results.columns:
    display_columns.append("Actual")
if "Correct" in results.columns:
    display_columns.append("Correct")

st.dataframe(results[display_columns], use_container_width=True)
