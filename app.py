import streamlit as st
import pickle
import sys
import os
import pandas as pd

# -------------------------------------------------
# Utility
# -------------------------------------------------
def safe_text(value):
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
from patient_risk_prediction.chd_inference import predict_chd_risk
from patient_risk_prediction.ckd_inference import predict_ckd_risk

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Disease Risk Assessment & Safe Drug Recommendation",
    layout="wide"
)

st.title("🩺 Disease Risk Assessment & Safe Drug Recommendation System")

st.write(
    "This system provides symptom-based disease risk assessment, "
    "drug recommendation with safety validation, and patient-data-based "
    "hypertension risk prediction."
)

st.info(
    "⚠️ Disclaimer: This system is a clinical decision-support tool and "
    "does not replace professional medical diagnosis."
)

# -------------------------------------------------
# Create Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Disease Prediction",
    "🧍 Hypertension Risk Prediction",
    "❤️ CHD Risk Prediction",
    "🧪 CKD Risk Prediction",
    "💊 Standalone DDI Checker"
])

# =================================================
# TAB 1: SYMPTOM-BASED DISEASE PREDICTION
# =================================================
with tab1:
    with open("disease_prediction/symptom_list.pkl", "rb") as f:
        symptom_list = pickle.load(f)

    st.subheader("Select Symptoms (Grouped for Better Experience)")

    BASE_CATEGORIES = {
        "🌡️ General & Systemic": [
            "fever","fatigue","chills","sweating","weight_loss","weight_gain",
            "loss_of_appetite","malaise","lethargy","restlessness","dehydration"
        ],
        "🤧 ENT & Respiratory": [
            "cough","sneezing","breathlessness","wheezing","runny_nose",
            "nasal_congestion","throat_irritation","phlegm",
            "sinus_pressure","chest_congestion"
        ],
        "❤️ Cardiovascular": [
            "chest_pain","palpitations","fast_heart_rate","slow_heart_rate",
            "cold_hands_and_feets","swelling_of_legs","fainting"
        ],
        "🤢 Gastrointestinal": [
            "nausea","vomiting","abdominal_pain","stomach_pain","acidity",
            "diarrhea","constipation","indigestion","bloating",
            "loss_of_taste","loss_of_smell"
        ],
        "🧠 Neurological": [
            "headache","dizziness","seizures","loss_of_balance",
            "tingling","numbness","confusion","blurred_vision","memory_loss"
        ],
        "🧠 Psychological": [
            "anxiety","depression","irritability","mood_swings",
            "insomnia","excessive_sleepiness"
        ],
        "🦴 Musculoskeletal": [
            "joint_pain","muscle_pain","muscle_weakness","stiff_neck",
            "back_pain","neck_pain","cramps"
        ],
        "🧴 Skin & Dermatological": [
            "itching","skin_rash","skin_peeling","red_spots",
            "ulcers_on_tongue","yellowish_skin","dark_urine",
            "bruising","pus_filled_pimples"
        ],
        "🚽 Urinary & Renal": [
            "burning_micturition","frequent_urination",
            "blood_in_urine","foul_smelling_urine"
        ],
        "🧬 Endocrine & Metabolic": [
            "increased_appetite","excessive_hunger",
            "excessive_thirst","polyuria","weight_fluctuations"
        ],
        "🦠 Infection & Others": [
            "swollen_glands","toxic_look_(typhos)",
            "patches_in_throat","high_fever",
            "pain_behind_the_eyes"
        ]
    }

    categorized = {cat: [] for cat in BASE_CATEGORIES}
    uncategorized = []

    for symptom in symptom_list:
        placed = False
        for cat, items in BASE_CATEGORIES.items():
            if symptom in items:
                categorized[cat].append(symptom)
                placed = True
                break
        if not placed:
            uncategorized.append(symptom)

    if uncategorized:
        categorized["📌 Other Symptoms"] = uncategorized

    selected_symptoms = []

    for category, symptoms in categorized.items():
        if symptoms:
            with st.expander(category):
                cols = st.columns(4)
                for i, symptom in enumerate(sorted(symptoms)):
                    with cols[i % 4]:
                        if st.checkbox(
                            symptom.replace("_", " ").title(),
                            key=f"sym_{symptom}"
                        ):
                            selected_symptoms.append(symptom)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            top_2_diseases, contributing_symptoms = predict_disease(selected_symptoms)

            st.subheader("🧠 Possible Conditions")
            for idx, (disease, prob) in enumerate(top_2_diseases, start=1):
                st.write(f"**{idx}. {disease} — {prob}%**")

            if top_2_diseases[0][1] < 50:
                st.warning("⚠️ Prediction confidence is moderate. Further tests are recommended.")

            st.subheader("🔍 Contributing Symptoms (from your selection)")
            if contributing_symptoms:
                for sym in contributing_symptoms:
                    st.write(f"- {sym.replace('_', ' ').title()}")
            else:
                st.write("No dominant symptom influence identified.")

            primary_disease = top_2_diseases[0][0]
            drugs = recommend_drugs(primary_disease)

            st.subheader("💊 Recommended Drugs")
            for drug in drugs:
                st.write(f"- {safe_text(drug)}")

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

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 0, 120, key="chd_age")
        height = st.number_input("Height (cm)", 0, 220)
        weight = st.number_input("Weight (kg)", 0, 200)
        ap_hi = st.number_input("Systolic BP (mmHg)", 0, 250)
        ap_lo = st.number_input("Diastolic BP (mmHg)", 0, 200)

    with col2:
        gender = st.selectbox("Gender", ["Select","Female","Male"])
        cholesterol_text = st.selectbox("Cholesterol Level", [
            "Select",
            "Normal (<200 mg/dL) – Low risk",
            "Above Normal (200–239 mg/dL) – Medium risk",
            "Well Above Normal (≥240 mg/dL) – High risk"
        ])
        glucose_text = st.selectbox("Glucose Level", [
            "Select",
            "Normal (<100 mg/dL) – Low risk",
            "Above Normal (100–125 mg/dL) – Medium risk",
            "Well Above Normal (≥126 mg/dL) – High risk"
        ])
        smoke = st.selectbox("Smoking Habit", ["Select","No","Yes"])
        alco = st.selectbox("Alcohol Intake", ["Select","No","Yes"])
        active = st.selectbox("Physically Active", ["Select","No","Yes"])

    if st.button("Predict Hypertension Risk"):
        if "Select" in [gender, cholesterol_text, glucose_text, smoke, alco, active]:
            st.warning("⚠️ Please fill all fields correctly")
        else:
            patient_data = {
                "gender": 1 if gender == "Female" else 2,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": {"Normal (<200 mg/dL) – Low risk":1,
                                "Above Normal (200–239 mg/dL) – Medium risk":2,
                                "Well Above Normal (≥240 mg/dL) – High risk":3}[cholesterol_text],
                "gluc": {"Normal (<100 mg/dL) – Low risk":1,
                         "Above Normal (100–125 mg/dL) – Medium risk":2,
                         "Well Above Normal (≥126 mg/dL) – High risk":3}[glucose_text],
                "smoke": 1 if smoke == "Yes" else 0,
                "alco": 1 if alco == "Yes" else 0,
                "active": 1 if active == "Yes" else 0,
                "age_years": age
            }

            risk_label, probability = predict_hypertension_risk(patient_data)
            if risk_label.startswith("High"):
                st.error(f"{risk_label} ({probability:.2f}%)")
            else:
                st.success(f"{risk_label} ({probability:.2f}%)")

# =================================================
# TAB 3: CHD RISK PREDICTION
# =================================================
with tab3:
    st.subheader("❤️ Coronary Heart Disease Risk Assessment")

    col1, col2 = st.columns(2)

    # -------- COLUMN 1 --------
    with col1:
        chd_age_input = st.number_input("Age (years)", 0, 120, key="chd_age_input_v2")
        chd_trestbps_input = st.number_input("Resting Blood Pressure (mmHg)", 0, 250, key="chd_bp_input_v2")
        chd_chol_input = st.number_input("Cholesterol (mg/dL)", 0, 600, key="chd_chol_input_v2")
        chd_thalach_input = st.number_input("Maximum Heart Rate Achieved", 0, 250, key="chd_thalach_input_v2")
        chd_oldpeak_input = st.number_input("ST Depression", 0.0, 10.0, key="chd_oldpeak_input_v2")

        chd_exang_text = st.selectbox(
            "Exercise Induced Angina",
            ["Select", "No", "Yes"],
            key="chd_exang_input_v2"
        )

    # -------- COLUMN 2 --------
    with col2:
        chd_sex_text = st.selectbox(
            "Sex",
            ["Select", "Male", "Female"],
            key="chd_sex_input_v2"
        )

        chd_cp_text = st.selectbox(
            "Chest Pain Type",
            ["Select", "Typical Angina", "Atypical Angina",
             "Non-anginal Pain", "Asymptomatic"],
            key="chd_cp_input_v2"
        )

        chd_restecg_text = st.selectbox(
            "Resting ECG Results",
            ["Select", "Normal", "ST-T Wave Abnormality",
             "Left Ventricular Hypertrophy"],
            key="chd_restecg_input_v2"
        )

        chd_fbs_text = st.selectbox(
            "Fasting Blood Sugar",
            ["Select", "Normal", "High"],
            key="chd_fbs_input_v2"
        )

        chd_slope_text = st.selectbox(
            "ST Segment Slope",
            ["Select", "Upsloping", "Flat", "Downsloping"],
            key="chd_slope_input_v2"
        )

        chd_ca_input = st.selectbox(
            "Major Vessels Colored (0–3)",
            [0, 1, 2, 3],
            key="chd_ca_input_v2"
        )

        chd_thal_text = st.selectbox(
            "Thalassemia Status",
            ["Select", "Normal", "Fixed Defect", "Reversible Defect"],
            key="chd_thal_input_v2"
        )

    if st.button("Predict CHD Risk", key="chd_predict_button_v2"):

        if "Select" in [
            chd_sex_text, chd_cp_text, chd_restecg_text,
            chd_fbs_text, chd_slope_text,
            chd_thal_text, chd_exang_text
        ]:
            st.warning("⚠️ Please complete all fields before prediction.")
        else:

            cp_map = {
                "Typical Angina": 0,
                "Atypical Angina": 1,
                "Non-anginal Pain": 2,
                "Asymptomatic": 3
            }

            restecg_map = {
                "Normal": 0,
                "ST-T Wave Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }

            slope_map = {
                "Upsloping": 0,
                "Flat": 1,
                "Downsloping": 2
            }

            thal_map = {
                "Normal": 1,
                "Fixed Defect": 2,
                "Reversible Defect": 3
            }

            input_data = {
                "age": chd_age_input,
                "sex": 1 if chd_sex_text == "Male" else 0,
                "cp": cp_map[chd_cp_text],
                "trestbps": chd_trestbps_input,
                "chol": chd_chol_input,
                "fbs": 1 if chd_fbs_text == "High" else 0,
                "restecg": restecg_map[chd_restecg_text],
                "thalachh": chd_thalach_input,
                "exang": 1 if chd_exang_text == "Yes" else 0,
                "oldpeak": chd_oldpeak_input,
                "slope": slope_map[chd_slope_text],
                "ca": chd_ca_input,
                "thal": thal_map[chd_thal_text]
            }

            result = predict_chd_risk(input_data)

            if result["risk_level"] == "High Risk":
                st.error(f"{result['decision']} ({result['risk_probability']}%)")
            elif result["risk_level"] == "Moderate Risk":
                st.warning(f"{result['decision']} ({result['risk_probability']}%)")
            else:
                st.success(f"{result['decision']} ({result['risk_probability']}%)")

# =================================================
# TAB 4: CKD RISK PREDICTION
# =================================================
with tab4:
    st.subheader("🧪 Chronic Kidney Disease Risk Assessment")

    col1, col2 = st.columns(2)

    # -------- COLUMN 1 --------
    with col1:
        ckd_age_input = st.number_input("Age (years)", 0, 120, key="ckd_age_input_v2")
        ckd_bp_input = st.number_input("Blood Pressure (mmHg)", 0, 200, key="ckd_bp_input_v2")
        ckd_bgr_input = st.number_input("Blood Glucose Random (mg/dL)", 0, 500, key="ckd_bgr_input_v2")
        ckd_bu_input = st.number_input("Blood Urea (mg/dL)", 0, 300, key="ckd_bu_input_v2")
        ckd_sc_input = st.number_input("Serum Creatinine (mg/dL)", 0.0, 20.0, key="ckd_sc_input_v2")
        ckd_sod_input = st.number_input("Sodium (mEq/L)", 0, 200, key="ckd_sod_input_v2")
        ckd_pot_input = st.number_input("Potassium (mEq/L)", 0.0, 10.0, key="ckd_pot_input_v2")
        ckd_hemo_input = st.number_input("Hemoglobin (gms)", 0.0, 20.0, key="ckd_hemo_input_v2")
        ckd_pcv_input = st.number_input("Packed Cell Volume (%)", 0, 60, key="ckd_pcv_input_v2")
        ckd_wc_input = st.number_input("White Blood Cell Count", 0, 20000, key="ckd_wc_input_v2")
        ckd_rc_input = st.number_input("Red Blood Cell Count (millions/cmm)", 0.0, 10.0, key="ckd_rc_input_v2")

    # -------- COLUMN 2 --------
    with col2:
        ckd_sg_text = st.selectbox("Specific Gravity",
                                   ["Select", 1.005, 1.010, 1.015, 1.020, 1.025],
                                   key="ckd_sg_input_v2")

        ckd_al = st.selectbox("Albumin Level (0–5)",
                              ["Select", 0,1,2,3,4,5],
                              key="ckd_al_input_v2")

        ckd_su = st.selectbox("Urine Sugar Level (0–5)",
                              ["Select", 0,1,2,3,4,5],
                              key="ckd_su_input_v2")

        ckd_rbc = st.selectbox("Red Blood Cells",
                               ["Select","normal","abnormal"],
                               key="ckd_rbc_input_v2")

        ckd_pc = st.selectbox("Pus Cell",
                              ["Select","normal","abnormal"],
                              key="ckd_pc_input_v2")

        ckd_pcc = st.selectbox("Pus Cell Clumps",
                               ["Select","notpresent","present"],
                               key="ckd_pcc_input_v2")

        ckd_ba = st.selectbox("Bacteria in Urine",
                              ["Select","notpresent","present"],
                              key="ckd_ba_input_v2")

        ckd_htn_text = st.selectbox("History of Hypertension",
                                    ["Select","No","Yes"],
                                    key="ckd_htn_input_v2")

        ckd_dm_text = st.selectbox("History of Diabetes",
                                   ["Select","No","Yes"],
                                   key="ckd_dm_input_v2")

        ckd_cad_text = st.selectbox("Coronary Artery Disease",
                                    ["Select","No","Yes"],
                                    key="ckd_cad_input_v2")

        ckd_appet_text = st.selectbox("Appetite",
                                      ["Select","Good","Poor"],
                                      key="ckd_appet_input_v2")

        ckd_pe_text = st.selectbox("Pedal Edema",
                                   ["Select","No","Yes"],
                                   key="ckd_pe_input_v2")

        ckd_ane_text = st.selectbox("Anemia",
                                    ["Select","No","Yes"],
                                    key="ckd_ane_input_v2")

    if st.button("Predict CKD Risk", key="ckd_predict_button_v2"):

        if "Select" in [
            ckd_sg_text, ckd_al, ckd_su, ckd_rbc, ckd_pc,
            ckd_pcc, ckd_ba, ckd_htn_text, ckd_dm_text,
            ckd_cad_text, ckd_appet_text, ckd_pe_text,
            ckd_ane_text
        ]:
            st.warning("⚠️ Please complete all fields before prediction.")
        else:
            input_data = {
                "age": ckd_age_input,
                "bp": ckd_bp_input,
                "sg": float(ckd_sg_text),
                "al": ckd_al,
                "su": ckd_su,
                "bgr": ckd_bgr_input,
                "bu": ckd_bu_input,
                "sc": ckd_sc_input,
                "sod": ckd_sod_input,
                "pot": ckd_pot_input,
                "hemo": ckd_hemo_input,
                "pcv": ckd_pcv_input,
                "wc": ckd_wc_input,
                "rc": ckd_rc_input,
                "rbc": ckd_rbc,
                "pc": ckd_pc,
                "pcc": ckd_pcc,
                "ba": ckd_ba,
                "htn": "yes" if ckd_htn_text == "Yes" else "no",
                "dm": "yes" if ckd_dm_text == "Yes" else "no",
                "cad": "yes" if ckd_cad_text == "Yes" else "no",
                "appet": "good" if ckd_appet_text == "Good" else "poor",
                "pe": "yes" if ckd_pe_text == "Yes" else "no",
                "ane": "yes" if ckd_ane_text == "Yes" else "no"
            }

            result = predict_ckd_risk(input_data)

            if result["risk_level"] == "High Risk":
                st.error(f"{result['decision']} ({result['risk_probability']}%)")
            elif result["risk_level"] == "Moderate Risk":
                st.warning(f"{result['decision']} ({result['risk_probability']}%)")
            else:
                st.success(f"{result['decision']} ({result['risk_probability']}%)")

# =================================================
# TAB 5: STANDALONE DDI CHECKER
# =================================================
with tab5:
    st.subheader("💊 Standalone Drug–Drug Interaction Checker")

    interaction_df = pd.read_csv("drug_safety/interaction_data.csv")
    all_drugs = sorted(set(interaction_df["drug1"]).union(set(interaction_df["drug2"])))

    selected_drugs = st.multiselect("Select two or more drugs", all_drugs)

    if st.button("Check Drug Interaction"):
        if len(selected_drugs) < 2:
            st.warning("⚠️ Please select at least two drugs")
        else:
            results = check_drug_safety(selected_drugs)
            for res in results:
                if res.startswith("⚠️"):
                    st.warning(res)
                else:
                    st.success(res)
