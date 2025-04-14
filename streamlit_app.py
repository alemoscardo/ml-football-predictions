import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load features list
features = joblib.load('models/features_used.pkl')

st.title("Football Match Outcome Predictor")

# Let user choose model
model_choice = st.selectbox("Select Prediction Model", ["Logistic Regression", "Random Forest" ])
model_path = (
    "models/logistic_regression_model.pkl" if model_choice == "Logistic Regression"
    else "models/random_forest_model.pkl"
)

model = joblib.load(model_path)
st.success(f"Loaded {model_choice} model successfully!")

# ==========================
# CSV Upload Section
# ==========================

st.header("Upload Match Stats (CSV)")

st.markdown("""
> ℹ️ **Need an example?**  
> A sample CSV is available in the [`data/` folder of this repository](https://github.com/alemoscardo/ml-football-predictions/tree/main/data) for testing the app.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Helper function to preprocess uploaded CSV
def preprocess_uploaded_csv(df_raw):
    required_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST',
                     'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    
    missing = [col for col in required_cols if col not in df_raw.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None

    df = df_raw.copy()

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

    df['HomeShotAcc'] = df['HomeShotsTarget'] / df['HomeShots'].replace(0, np.nan)
    df['AwayShotAcc'] = df['AwayShotsTarget'] / df['AwayShots'].replace(0, np.nan)
    df['AggressionDiff'] = (
        df['HomeFouls'] + df['HomeYellows'] * 2 + df['HomeReds'] * 3
    ) - (
        df['AwayFouls'] + df['AwayYellows'] * 2 + df['AwayReds'] * 3
    )
    df['HomeGoalDiff'] = df['HomeGoals'] - df['AwayGoals']

    return df

# Process and predict
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df_raw.head())

    df_processed = preprocess_uploaded_csv(df_raw)

    if df_processed is not None:
        df_ready = df_processed.dropna(subset=features)
        X = df_ready[features]

        preds = model.predict(X)

        label_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        df_ready['Predicted'] = [label_map.get(p, p) for p in preds]

        if 'FTR' in df_raw.columns:
            df_ready['Actual'] = df_raw['FTR'].map(label_map)

        st.subheader("Prediction Results")
        display_cols = ['HomeTeam', 'AwayTeam', 'Predicted']
        if 'Actual' in df_ready.columns:
            display_cols.append('Actual')

        st.dataframe(df_ready[display_cols])

        # ----------------------------------------
        # ✅ Accuracy summary (if actual results exist)
        # ----------------------------------------
        if 'Actual' in df_ready.columns:
            correct_preds = (df_ready['Predicted'] == df_ready['Actual']).sum()
            total_preds = len(df_ready)
            accuracy = correct_preds / total_preds * 100

            st.markdown(f"**Accuracy:** {accuracy:.2f}% ({correct_preds} out of {total_preds} correct)")

            # ----------------------------------------
            # 📊 Bar Chart of Predictions vs. Actual
            # ----------------------------------------
            st.subheader("Prediction vs Actual Result Distribution")

            counts = pd.DataFrame({
                'Predicted': df_ready['Predicted'].value_counts(),
                'Actual': df_ready['Actual'].value_counts()
            }).fillna(0).astype(int)

            st.bar_chart(counts)


# ==========================
# Manual Input Section
# ==========================

st.markdown("---")
st.header("Enter Match Stats Manually")

input_data = {}

for feature in features:
    if 'Acc' in feature:
        input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)
    elif 'Diff' in feature:
        input_data[feature] = st.slider(feature, -10.0, 10.0, 0.0)
    else:
        input_data[feature] = st.number_input(feature, min_value=0, max_value=40, value=0)

if st.button("Predict Outcome"):
    X_manual = pd.DataFrame([input_data])[features]
    pred = model.predict(X_manual)[0]
    label_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    st.subheader("Prediction Result")
    st.write(f"**{label_map.get(pred, pred)}**")
