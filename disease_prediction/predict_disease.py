import pickle
import pandas as pd
import numpy as np

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
with open("disease_prediction/disease_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load symptom list
with open("disease_prediction/symptom_list.pkl", "rb") as f:
    symptom_list = pickle.load(f)


def predict_disease(selected_symptoms):
    """
    selected_symptoms: list of symptom names selected by user

    returns:
    - top_2_diseases: list of tuples -> [(disease, probability), ...]
    - contributing_symptoms: list of symptoms from user input that influenced prediction
    """

    # -------------------------------------------------
    # Create input dataframe (binary symptom vector)
    # -------------------------------------------------
    input_data = pd.DataFrame(
        [[1 if symptom in selected_symptoms else 0 for symptom in symptom_list]],
        columns=symptom_list
    )

    # -------------------------------------------------
    # Predict probabilities
    # -------------------------------------------------
    probabilities = model.predict_proba(input_data)[0]
    class_labels = model.classes_

    # -------------------------------------------------
    # Get top-2 predictions
    # -------------------------------------------------
    top_indices = np.argsort(probabilities)[::-1][:2]

    top_2_diseases = [
        (class_labels[i], round(probabilities[i] * 100, 2))
        for i in top_indices
    ]

    # -------------------------------------------------
    # Explainability (ONLY user-selected symptoms)
    # -------------------------------------------------
    feature_importance = model.feature_importances_

    symptom_importance_map = {
        symptom: feature_importance[idx]
        for idx, symptom in enumerate(symptom_list)
        if symptom in selected_symptoms
    }

    # Sort by importance (descending)
    contributing_symptoms = sorted(
        symptom_importance_map,
        key=symptom_importance_map.get,
        reverse=True
    )

    return top_2_diseases, contributing_symptoms
