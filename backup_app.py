import streamlit as st
import pickle
import sys
import os
import pandas as pd

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from disease_prediction.predict_disease import predict_disease
from drug_recommendation.recommend_drug import recommend_drugs
from drug_safety.safety_check import check_drug_safety

# Page config
st.set_page_config(
    page_title="Disease Risk Assessment & Safe Drug Recommendation",
    layout="wide"
)

st.title("🩺 Disease Risk Assessment & Safe Drug Recommendation System")
st.write(
    "This system predicts disease risk based on symptoms, "
    "recommends indicative drugs, and validates drug safety. "
    "It also provides a standalone drug–drug interaction checker."
)

# ===============================
# Create Tabs
# ===============================
tab1, tab2 = st.tabs(["🩺 Disease Prediction", "💊 Standalone DDI Checker"])

# =========================================================
# TAB 1: EXISTING SYSTEM (UNCHANGED LOGIC)
# =========================================================
with tab1:
    # Load symptom list
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
        if len(selected_symptoms) == 0:
            st.warning("⚠️ Please select at least one symptom")
        else:
            # Module 1: Disease Prediction
            disease = predict_disease(selected_symptoms)
            st.success(f"🧠 Predicted Disease: **{disease}**")

            # Module 2: Drug Recommendation
            drugs = recommend_drugs(disease)

            st.subheader("💊 Recommended Drugs")
            for drug in drugs:
                st.write(f"- {drug}")

            # Module 3: Drug Safety Validation
            st.subheader("🛡️ Drug Safety Validation")
            safety_messages = check_drug_safety(drugs)

            for msg in safety_messages:
                if "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.success(msg)

# =========================================================
# TAB 2: STANDALONE DRUG–DRUG INTERACTION CHECKER
# =========================================================
with tab2:
    st.subheader("💊 Standalone Drug–Drug Interaction Checker")
    st.write(
        "This tool allows users to independently check potential "
        "drug–drug interactions without disease prediction."
    )

    # Load interaction data
    interaction_df = pd.read_csv("drug_safety/interaction_data.csv")

    all_drugs = sorted(
        set(interaction_df["drug1"]).union(set(interaction_df["drug2"]))
    )

    selected_drugs = st.multiselect(
        "Select two or more drugs",
        all_drugs
    )

    if st.button("Check Drug Interaction"):
        if len(selected_drugs) < 2:
            st.warning("⚠️ Please select at least two drugs")
        else:
            results = check_drug_safety(selected_drugs)

            st.subheader("🛡️ Interaction Results")
            for res in results:
                if "⚠️" in res:
                    st.warning(res)
                else:
                    st.success(res)
