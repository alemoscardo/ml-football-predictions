from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Load model using absolute path relative to script
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / 'models' / 'logistic_regression_model.pkl')
features = joblib.load(BASE_DIR / 'models' / 'features_used.pkl')


# Sample input
input_data = {
    'HomeShots': 14,
    'AwayShots': 9,
    'HomeShotsTarget': 7,
    'AwayShotsTarget': 3,
    'HomeCorners': 6,
    'AwayCorners': 4,
    'HomeFouls': 12,
    'AwayFouls': 15,
    'HomeYellows': 2,
    'AwayYellows': 1,
    'HomeReds': 0,
    'AwayReds': 1,
    'HomeGoalDiff': 1,       # e.g., 2-1 = +1
    'HomeShotAcc': 0.5,      # 7/14
    'AwayShotAcc': 0.33,     # 3/9
    'AggressionDiff': (12 + 2*2 + 3*0) - (15 + 2*1 + 3*1)
}

# Convert to model input array (in correct order)
X = pd.DataFrame([input_data])[features]

# Predict
pred = model.predict(X)[0]

# Output result
result_map = {'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'}
print(f"Prediction: {pred} → {result_map.get(pred, pred)}")
