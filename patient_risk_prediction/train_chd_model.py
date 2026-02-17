import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/heart.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# ===============================
# 2. Features & Target
# ===============================
X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# 3. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. Train High-Performance RF
# ===============================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================
# 5. Evaluate
# ===============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n✅ CHD Model Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 6. Save Model
# ===============================
with open("models/chd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ CHD model saved successfully at models/chd_model.pkl")

# Save feature order for inference consistency
with open("models/chd_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("CHD feature order saved.")
