import pickle
import pandas as pd
import numpy as np

# Load model
with open("models/ckd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature order
with open("models/ckd_features.pkl", "rb") as f:
    feature_order = pickle.load(f)

# Load label encoders
with open("models/ckd_label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)


def predict_ckd_risk(input_data):

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col])

    # Ensure feature order matches training
    df = df[feature_order]

    # Predict probability
    prob = model.predict_proba(df)[0][1] * 100

    # Risk level
    if prob < 40:
        level = "Low Risk"
        decision = "CKD Unlikely"
    elif prob < 70:
        level = "Moderate Risk"
        decision = "CKD Possible"
    else:
        level = "High Risk"
        decision = "CKD Likely"

    return {
        "risk_probability": round(prob, 2),
        "risk_level": level,
        "decision": decision
    }
