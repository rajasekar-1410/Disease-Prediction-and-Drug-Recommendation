import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/Training.csv")

# Separate features and target
X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# FIX: Replace NaN values with 0
X = X.fillna(0)


# Save symptom list
symptom_list = X.columns.tolist()
with open("disease_prediction/symptom_list.pkl", "wb") as f:
    pickle.dump(symptom_list, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# Save model
with open("disease_prediction/disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Module 1 training completed successfully")
