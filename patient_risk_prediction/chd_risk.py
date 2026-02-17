import pickle

# Load trained CHD model (we’ll train if not yet available)
with open("models/chd_model.pkl", "rb") as f:
    model = pickle.load(f)

def get_risk_level(prob):
    if prob < 30:
        return "Low Risk"
    elif prob < 60:
        return "Moderate Risk"
    else:
        return "High Risk"

def predict_chd_risk(input_data):
    prob = model.predict_proba(input_data)[0][1] * 100
    level = get_risk_level(prob)

    decision = "Likely" if prob >= 60 else "Unlikely"

    return {
        "probability": round(prob, 2),
        "risk_level": level,
        "decision": decision
    }
