import pickle
import pandas as pd

# Load trained model
with open("disease_prediction/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load symptom list
with open("disease_prediction/symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)

def predict_disease(selected_symptoms):
    """
    selected_symptoms: list of symptom names selected by user
    returns: predicted disease name
    """

    # Create input as DataFrame with correct feature names
    input_data = pd.DataFrame(
        [[1 if symptom in selected_symptoms else 0 for symptom in symptom_list]],
        columns=symptom_list
    )

    # Predict disease
    prediction = model.predict(input_data)[0]
    return prediction
