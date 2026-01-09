
# 🩺 Disease Risk Assessment & Safe Drug Recommendation System

A **hybrid machine learning–based healthcare decision support system** that predicts diseases from symptoms, assesses hypertension risk using patient clinical data, recommends indicative drugs, and validates drug–drug safety interactions through an interactive web interface.

---

## 📌 Project Overview

Healthcare decision-making often requires analyzing multiple sources of information such as patient symptoms, clinical measurements, medication choices, and drug safety. Existing systems usually focus on only one task.

This project proposes an **integrated system** combining:
- Symptom-based disease prediction
- Patient-data-based hypertension risk assessment
- Drug recommendation
- Drug–drug interaction (DDI) safety validation

All modules are deployed using a **Streamlit web application**.

---

## 🎯 Objectives
- Predict diseases based on symptoms using ML
- Assess hypertension risk using patient clinical data
- Recommend indicative drugs for diseases
- Validate drug–drug interactions
- Provide an interactive and explainable UI

---

## 🧠 System Modules

### 🔹 Module 1: Symptom-Based Disease Prediction
- Model: Random Forest Classifier
- Input: Patient symptoms
- Output: Predicted disease

### 🔹 Module 2: Drug Recommendation
- Input: Predicted disease
- Output: Indicative drugs

### 🔹 Module 3: Drug Safety Validation (DDI)
- Input: Drug list
- Output: Interaction warnings or safe confirmation

### 🔹 Module 4: Hypertension Risk Prediction
- Input: Patient clinical data
- Output: High / Low risk with probability

### 🔹 Module 5: Standalone DDI Checker
- Independent drug–drug interaction validation

---

## 🛠️ Technologies Used
- Python
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Git & GitHub

---

## 📊 Datasets Used
- Symptom–Disease dataset
- Cardiovascular (Hypertension) dataset
- Drug–Disease mapping dataset
- Drug–Drug interaction dataset

---

## 🚀 How to Run

```bash
git clone https://github.com/rajasekar-1410/Disease-Prediction-and-Drug-Recommendation.git
cd Disease-Prediction-and-Drug-Recommendation
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Disclaimer
This application is for **academic and educational purposes only** and does not replace professional medical advice.

---

## 🔮 Future Enhancements
- More diseases
- Real-time medical APIs
- Explainable AI
- Cloud deployment

---


