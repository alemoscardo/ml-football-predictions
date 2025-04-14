import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and features
model = joblib.load('models/random_forest_model.pkl')
features = joblib.load('models/features_used.pkl')

st.title("⚽ Football Match Outcome Predictor")

# ========================================
# CSV Upload Section
# ========================================

st.markdown("## 📂 Upload a CSV of Match Stats")

uploaded_file = st.file_uploader("Upload your match stats CSV", type=["csv"])

# Helper function: Convert raw Football-Data format to model features
def preprocess_uploaded_csv(df_raw):
    required_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
                     'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    missing = [col for col in required_cols if col not in df_raw.columns]
    if missing:
        st.error(f"Missing columns in uploaded file: {missing}")
        return None

    df = df_raw.copy()

    # Rename
    df.rename(columns={
        'FTHG': 'HomeGoals',
        'FTAG': 'AwayGoals',
        'HS': 'HomeShots', 'AS': 'AwayShots',
        'HST': 'HomeShotsTarget', 'AST': 'AwayShotsTarget',
        'HC': 'HomeCorners', 'AC': 'AwayCorners',
        'HF': 'HomeFouls', 'AF': 'AwayFouls',
        'HY': 'HomeYellows', 'AY': 'AwayYellows',
        'HR': 'HomeReds', 'AR': 'AwayReds',
    }, inplace=True)

    # Feature Engineering
    df['HomeShotAcc'] = df['HomeShotsTarget'] / df['HomeShots'].replace(0, np.nan)
    df['AwayShotAcc'] = df['AwayShotsTarget'] / df['AwayShots'].replace(0, np.nan)
    df['AggressionDiff'] = (
        df['HomeFouls'] + df['HomeYellows'] * 2 + df['HomeReds'] * 3
    ) - (
        df['AwayFouls'] + df['AwayYellows'] * 2 + df['AwayReds'] * 3
    )
    df['HomeGoalDiff'] = df['HomeGoals'] - df['AwayGoals']

    return df

# Process and predict if file uploaded
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df_raw.head())

    df_processed = preprocess_uploaded_csv(df_raw)

    if df_processed is not None:
        df_ready = df_processed.dropna(subset=features)
        X = df_ready[features]

        preds = model.predict(X)

        result_map = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'}

        # Prediction
        df_ready['Prediction'] = preds
        df_ready['PredictionLabel'] = df_ready['Prediction'].map(result_map)

        # If actual result exists in data
        if 'FTR' in df_raw.columns:
            df_ready['ActualResult'] = df_raw['FTR'].map(result_map)
        else:
            df_ready['ActualResult'] = '❓'

        # Display
        st.success("✅ Predictions complete!")

        st.dataframe(
            df_ready[['HomeTeam', 'AwayTeam', 'PredictionLabel', 'ActualResult']]
            .rename(columns={
                'PredictionLabel': '🔮 Predicted',
                'ActualResult': '✅ Real Result'
            })
        )


# ========================================
# Manual Input Section
# ========================================

st.markdown("---")
st.markdown("## 🧪 Or Enter Match Stats Manually")

input_data = {}

# Create sliders or number inputs for each feature
for feature in features:
    if 'Acc' in feature:
        input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)
    elif 'Diff' in feature:
        input_data[feature] = st.slider(feature, -10.0, 10.0, 0.0)
    else:
        input_data[feature] = st.number_input(feature, min_value=0, max_value=40, value=10)

if st.button("Predict Match Outcome"):
    X_manual = pd.DataFrame([input_data])[features]
    pred = model.predict(X_manual)[0]

    result_map = {
        'H': '🏠 Home Win',
        'D': '🤝 Draw',
        'A': '✈️ Away Win'
    }

    st.markdown(f"### 🧠 Prediction: **{result_map.get(pred, pred)}**")
