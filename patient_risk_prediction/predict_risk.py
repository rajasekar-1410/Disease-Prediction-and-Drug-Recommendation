import pickle
import pandas as pd

# Load model
with open("patient_risk_prediction/hypertension_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("patient_risk_prediction/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Feature order (VERY IMPORTANT)
FEATURES = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years"
]


def predict_hypertension_risk(input_data: dict):
    """
    input_data: dictionary with patient features
    returns: (risk_label: str, probability: float)
    """

    # Create DataFrame with fixed feature order
    df = pd.DataFrame([input_data], columns=FEATURES)

    # Scale features
    df_scaled = scaler.transform(df)

    # Model prediction
    prediction = int(model.predict(df_scaled)[0])
    probability = float(model.predict_proba(df_scaled)[0][1])

    if prediction == 1:
        return "High Hypertension Risk", round(probability * 100, 2)
    else:
        return "Low Hypertension Risk", round(probability * 100, 2)
