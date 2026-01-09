import pandas as pd

# Load dataset
df = pd.read_csv("patient_risk_prediction/hypertension_data.csv", sep=';')

# -------------------------------
# Convert age from days to years
# -------------------------------
df['age_years'] = (df['age'] / 365).astype(int)
df.drop(columns=['age'], inplace=True)

# -------------------------------
# Create Hypertension Risk Label
# -------------------------------
df['hypertension'] = ((df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)).astype(int)

# -------------------------------
# Drop unused target column
# -------------------------------
df.drop(columns=['cardio'], inplace=True)

# -------------------------------
# Save processed dataset
# -------------------------------
df.to_csv("patient_risk_prediction/hypertension_processed.csv", index=False)

print("✅ Preprocessing completed")
print("Dataset shape:", df.shape)
print(df['hypertension'].value_counts())
