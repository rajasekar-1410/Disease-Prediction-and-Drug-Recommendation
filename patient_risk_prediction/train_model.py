import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Load processed dataset
# -------------------------------
df = pd.read_csv("patient_risk_prediction/hypertension_processed.csv")

X = df.drop(["hypertension", "id"], axis=1)
y = df["hypertension"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Feature scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# Train Logistic Regression
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Hypertension Risk Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model & scaler
# -------------------------------
with open("patient_risk_prediction/hypertension_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("patient_risk_prediction/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully")
