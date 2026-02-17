import pandas as pd
import pickle

# Load trained model
with open("models/chd_model.pkl", "rb") as f:
    chd_model = pickle.load(f)

# Load feature order
with open("models/chd_features.pkl", "rb") as f:
    chd_features = pickle.load(f)


def get_risk_level(prob):
    if prob < 30:
        return "Low Risk"
    elif prob < 60:
        return "Moderate Risk"
    else:
        return "High Risk"


def predict_chd_risk(input_dict):
    # Create DataFrame with correct feature names
    input_df = pd.DataFrame([[input_dict[f] for f in chd_features]],
                            columns=chd_features)

    # Predict probability
    risk_prob = chd_model.predict_proba(input_df)[0][1] * 100

    risk_level = get_risk_level(risk_prob)
    decision = "Likely" if risk_prob >= 60 else "Unlikely"

    return {
        "risk_probability": round(risk_prob, 2),
        "risk_level": risk_level,
        "decision": decision
    }
