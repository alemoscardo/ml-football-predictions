import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import (
    MODEL_FEATURES,
    OPTIONAL_ODDS_COLUMNS,
    REQUIRED_MATCH_STATS,
    RENAME_MAP,
    build_feature_matrix,
    prepare_matches_dataframe,
)

st.set_page_config(page_title="Football Match Outcome Predictor", layout="wide")

LABEL_MAP = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
MODEL_OPTIONS = {
    "Extra Trees (Tuned)": "models/extra_trees_model.pkl",
    "Logistic Regression (Tuned)": "models/logistic_regression_model.pkl",
}

try:
    features = joblib.load("models/features_used.pkl")
except Exception:
    features = MODEL_FEATURES

st.title("Football Match Outcome Predictor")
st.caption("Models trained with engineered match stats plus bookmaker odds features.")

model_choice = st.selectbox("Select Prediction Model", list(MODEL_OPTIONS.keys()))
model_path = MODEL_OPTIONS[model_choice]
model = joblib.load(model_path)
st.success(f"Loaded {model_choice} model successfully.")

renamed_to_raw = {value: key for key, value in RENAME_MAP.items()}


def find_missing_required_columns(df_raw: pd.DataFrame) -> list[str]:
    missing = []
    for required in REQUIRED_MATCH_STATS:
        raw_alias = renamed_to_raw.get(required)
        if required not in df_raw.columns and raw_alias not in df_raw.columns:
            missing.append(raw_alias or required)
    return missing


def find_missing_optional_odds(df_raw: pd.DataFrame) -> list[str]:
    missing = []
    for optional in OPTIONAL_ODDS_COLUMNS:
        raw_alias = renamed_to_raw.get(optional)
        if optional not in df_raw.columns and raw_alias not in df_raw.columns:
            missing.append(raw_alias or optional)
    return missing


st.header("Upload Match Stats (CSV)")
st.markdown(
    """
> Need an example?
> A sample CSV is available in the [`data/` folder](https://github.com/alemoscardo/ml-football-predictions/tree/main/data).
"""
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_raw.head())

    missing_required = find_missing_required_columns(df_raw)
    if missing_required:
        st.error(f"Missing required match-stat columns: {missing_required}")
    else:
        missing_odds = find_missing_optional_odds(df_raw)
        if missing_odds:
            st.info(
                "Optional odds columns are missing. Predictions will still run using median imputation "
                f"for {missing_odds}."
            )

        df_processed = prepare_matches_dataframe(df_raw)
        X = build_feature_matrix(df_raw, features)

        preds = model.predict(X)
        pred_labels = pd.Series(preds).map(LABEL_MAP).fillna(pd.Series(preds))

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(X).max(axis=1)

        results = df_processed.copy()
        results["Predicted"] = pred_labels.values

        if confidence is not None:
            confidence_pct = pd.Series(confidence, index=results.index).mul(100).round(1)
            results["Confidence"] = confidence_pct.map(lambda value: f"{value:.1f}%")

        if "Result" in df_processed.columns:
            results["Actual"] = df_processed["Result"].map(LABEL_MAP)

        st.subheader("Prediction Results")
        display_cols = [c for c in ["Date", "HomeTeam", "AwayTeam"] if c in results.columns]
        display_cols.append("Predicted")
        if "Confidence" in results.columns:
            display_cols.append("Confidence")
        if "Actual" in results.columns:
            display_cols.append("Actual")

        st.dataframe(results[display_cols])

        if "Result" in df_processed.columns:
            mask = df_processed["Result"].notna()
            if mask.any():
                accuracy = (preds[mask] == df_processed.loc[mask, "Result"].to_numpy()).mean() * 100
                st.markdown(f"**Accuracy:** {accuracy:.2f}%")

                st.subheader("Prediction vs Actual Distribution")
                counts = pd.DataFrame(
                    {
                        "Predicted": pd.Series(preds[mask]).map(LABEL_MAP).value_counts(),
                        "Actual": df_processed.loc[mask, "Result"].map(LABEL_MAP).value_counts(),
                    }
                ).fillna(0)
                st.bar_chart(counts)

st.markdown("---")
st.header("Enter Match Stats Manually")
st.caption("Bookmaker odds are optional but improve model quality.")

with st.form("manual_prediction_form"):
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        home_shots = st.number_input("Home Shots", min_value=0, max_value=50, value=12)
        home_shots_target = st.number_input("Home Shots on Target", min_value=0, max_value=30, value=5)
        home_corners = st.number_input("Home Corners", min_value=0, max_value=20, value=6)
        home_fouls = st.number_input("Home Fouls", min_value=0, max_value=40, value=11)
        home_yellows = st.number_input("Home Yellow Cards", min_value=0, max_value=10, value=2)
        home_reds = st.number_input("Home Red Cards", min_value=0, max_value=3, value=0)

    with col_mid:
        away_shots = st.number_input("Away Shots", min_value=0, max_value=50, value=10)
        away_shots_target = st.number_input("Away Shots on Target", min_value=0, max_value=30, value=4)
        away_corners = st.number_input("Away Corners", min_value=0, max_value=20, value=5)
        away_fouls = st.number_input("Away Fouls", min_value=0, max_value=40, value=12)
        away_yellows = st.number_input("Away Yellow Cards", min_value=0, max_value=10, value=2)
        away_reds = st.number_input("Away Red Cards", min_value=0, max_value=3, value=0)

    with col_right:
        use_odds = st.checkbox("Include bookmaker odds", value=True)
        b365_home = st.number_input("B365 Home Odds", min_value=1.01, max_value=50.0, value=2.1, step=0.01)
        b365_draw = st.number_input("B365 Draw Odds", min_value=1.01, max_value=50.0, value=3.3, step=0.01)
        b365_away = st.number_input("B365 Away Odds", min_value=1.01, max_value=50.0, value=3.4, step=0.01)

    predict_clicked = st.form_submit_button("Predict Outcome")

if predict_clicked:
    manual_row = {
        "HomeShots": home_shots,
        "AwayShots": away_shots,
        "HomeShotsTarget": home_shots_target,
        "AwayShotsTarget": away_shots_target,
        "HomeCorners": home_corners,
        "AwayCorners": away_corners,
        "HomeFouls": home_fouls,
        "AwayFouls": away_fouls,
        "HomeYellows": home_yellows,
        "AwayYellows": away_yellows,
        "HomeReds": home_reds,
        "AwayReds": away_reds,
    }

    if use_odds:
        manual_row["B365H"] = b365_home
        manual_row["B365D"] = b365_draw
        manual_row["B365A"] = b365_away
    else:
        manual_row["B365H"] = np.nan
        manual_row["B365D"] = np.nan
        manual_row["B365A"] = np.nan

    manual_df = pd.DataFrame([manual_row])
    X_manual = build_feature_matrix(manual_df, features)

    pred = model.predict(X_manual)[0]
    st.subheader("Prediction Result")
    st.write(f"**{LABEL_MAP.get(pred, pred)}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_manual)[0]
        proba_df = pd.DataFrame(
            {
                "Outcome": [LABEL_MAP[label] for label in model.classes_],
                "Probability": proba,
            }
        )
        st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}))
