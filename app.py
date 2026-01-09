import streamlit as st
import pickle
import sys
import os
import pandas as pd

def safe_text(value):
    """
    HARD convert any value into a plain Python string
    """
    try:
        if value is None:
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join([str(v) for v in value])
        if isinstance(value, dict):
            return str(value)
        return str(value)
    except Exception:
        return ""

# -------------------------------------------------
# Allow importing from project root
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from disease_prediction.predict_disease import predict_disease
from drug_recommendation.recommend_drug import recommend_drugs
from drug_safety.safety_check import check_drug_safety
from patient_risk_prediction.predict_risk import predict_hypertension_risk

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Disease Risk Assessment & Safe Drug Recommendation",
    layout="wide"
)

st.title("🩺 Disease Risk Assessment & Safe Drug Recommendation System")

st.write(
    "This system predicts disease risk based on symptoms, recommends indicative drugs, "
    "validates drug safety, and provides patient-data-based hypertension risk assessment, "
    "along with a standalone drug–drug interaction checker."
)

st.info(
    "⚠️ Disclaimer: This application is for educational and risk-assessment purposes only "
    "and does not replace professional medical diagnosis or treatment."
)

# -------------------------------------------------
# Create Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🩺 Disease Prediction",
    "🧍 Hypertension Risk Prediction",
    "💊 Standalone DDI Checker"
])

# =================================================
# TAB 1: SYMPTOM-BASED DISEASE PREDICTION
# =================================================
with tab1:
    with open("disease_prediction/symptom_list.pkl", "rb") as f:
        symptom_list = pickle.load(f)

    st.subheader("Select Symptoms")

    selected_symptoms = []
    cols = st.columns(4)

    for i, symptom in enumerate(symptom_list):
        with cols[i % 4]:
            if st.checkbox(symptom, key=f"sym_{i}"):
                selected_symptoms.append(symptom)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            disease = predict_disease(selected_symptoms)
            st.success(f"🧠 Predicted Disease: {disease}")

            drugs = recommend_drugs(disease)

            st.subheader("💊 Recommended Drugs")
            for drug in drugs:
                st.markdown(f"- {safe_text(drug)}")


            st.subheader("🛡️ Drug Safety Validation")
            safety_messages = check_drug_safety(drugs)

            for msg in safety_messages:
                msg_text = safe_text(msg)
                if msg_text.startswith("⚠️"):
                    st.warning(msg_text)
                else:
                    st.success(msg_text)


# =================================================
# TAB 2: HYPERTENSION RISK PREDICTION
# =================================================
with tab2:
    st.subheader("🧍 Hypertension Risk Prediction (Patient Data)")
    st.write("Enter patient clinical details. All fields are required.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120)
        height = st.number_input("Height (cm)", min_value=0, max_value=220)
        weight = st.number_input("Weight (kg)", min_value=0, max_value=200)
        ap_hi = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0, max_value=250)
        ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=200)

    with col2:
        gender = st.selectbox("Gender", ["Select", "Female", "Male"])

        cholesterol_text = st.selectbox(
            "Cholesterol Level",
            [
                "Select",
                "Normal (<200 mg/dL) – Low risk",
                "Above Normal (200–239 mg/dL) – Medium risk",
                "Well Above Normal (≥240 mg/dL) – High risk"
            ]
        )

        glucose_text = st.selectbox(
            "Glucose Level",
            [
                "Select",
                "Normal (<100 mg/dL) – Low risk",
                "Above Normal (100–125 mg/dL) – Medium risk",
                "Well Above Normal (≥126 mg/dL) – High risk"
            ]
        )

        smoke = st.selectbox("Smoking Habit", ["Select", "No", "Yes"])
        alco = st.selectbox("Alcohol Intake", ["Select", "No", "Yes"])
        active = st.selectbox("Physically Active", ["Select", "No", "Yes"])

    if st.button("Predict Hypertension Risk"):
        if (
            age == 0 or height == 0 or weight == 0 or
            ap_hi == 0 or ap_lo == 0 or
            gender == "Select" or
            cholesterol_text == "Select" or
            glucose_text == "Select" or
            smoke == "Select" or
            alco == "Select" or
            active == "Select"
        ):
            st.warning("⚠️ Please fill in all fields correctly")
            st.stop()

        if ap_lo >= ap_hi:
            st.warning("⚠️ Diastolic BP should be lower than Systolic BP")
            st.stop()

        patient_data = {
            "gender": 1 if gender == "Female" else 2,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": {
                "Normal (<200 mg/dL) – Low risk": 1,
                "Above Normal (200–239 mg/dL) – Medium risk": 2,
                "Well Above Normal (≥240 mg/dL) – High risk": 3
            }[cholesterol_text],
            "gluc": {
                "Normal (<100 mg/dL) – Low risk": 1,
                "Above Normal (100–125 mg/dL) – Medium risk": 2,
                "Well Above Normal (≥126 mg/dL) – High risk": 3
            }[glucose_text],
            "smoke": 1 if smoke == "Yes" else 0,
            "alco": 1 if alco == "Yes" else 0,
            "active": 1 if active == "Yes" else 0,
            "age_years": age
        }

        risk_label, probability = predict_hypertension_risk(patient_data)

        # Force native Python types (extra safety)
        label_text = str(risk_label)
        prob_text = f"{float(probability):.2f}"

        st.subheader("🧠 Prediction Result")
  
        if label_text.startswith("High"):
            st.error(f"{label_text} ({prob_text}%)")
        else:
            st.success(f"{label_text} ({prob_text}%)")


# =================================================
# TAB 3: STANDALONE DRUG–DRUG INTERACTION CHECKER
# =================================================
with tab3:
    st.subheader("💊 Standalone Drug–Drug Interaction Checker")
    st.write("Check potential drug–drug interactions independently.")

    # Load interaction data safely (NO direct rendering)
    try:
        interaction_df = pd.read_csv(os.path.join("drug_safety", "interaction_data.csv"))
        all_drugs = sorted(set(interaction_df["drug1"]).union(set(interaction_df["drug2"])))
    except Exception:
        st.error("⚠️ Drug interaction data could not be loaded")
        st.stop()

    selected_drugs = st.multiselect("Select two or more drugs", all_drugs)

    if st.button("Check Drug Interaction"):
        if len(selected_drugs) < 2:
            st.warning("⚠️ Please select at least two drugs")
        else:
            results = check_drug_safety(selected_drugs)

            if not isinstance(results, list):
                st.error("⚠️ Internal error in drug interaction module")
                st.stop()

            st.subheader("🛡️ Interaction Results")
            for res in results:
                res_text = safe_text(res)
                if res_text.startswith("⚠️"):
                    st.warning(res_text)
                else:
                    st.success(res_text)