import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and features
model = joblib.load('models/random_forest_model.pkl')
features = joblib.load('models/features_used.pkl')

st.title("⚽ Football Match Outcome Predictor")

st.markdown("### Enter Match Stats")

input_data = {}

# Create sliders or number inputs for each feature
for feature in features:
    if 'Acc' in feature:
        input_data[feature] = st.slider(feature, 0.0, 1.0, 0.5)
    elif 'Diff' in feature:
        input_data[feature] = st.slider(feature, -10.0, 10.0, 0.0)
    else:
        input_data[feature] = st.number_input(feature, min_value=0, max_value=40, value=10)

# Predict button
if st.button("Predict Match Outcome"):
    X = pd.DataFrame([input_data])[features]
    pred = model.predict(X)[0]

    result_map = {
        'H': '🏠 Home Win',
        'D': '🤝 Draw',
        'A': '✈️ Away Win'
    }

    st.markdown(f"### 🧠 Prediction: **{result_map.get(pred, pred)}**")
