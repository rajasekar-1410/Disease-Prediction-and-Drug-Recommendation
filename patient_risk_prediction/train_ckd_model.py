import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/kidney.csv")
print("Original Shape:", df.shape)

# ==============================
# 2. Drop ID column
# ==============================
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# ==============================
# 3. Clean Target Column
# ==============================
df["classification"] = df["classification"].astype(str).str.strip().str.lower()
df["classification"] = df["classification"].replace({
    "ckd": 1,
    "ckd\t": 1,
    "notckd": 0
})

df["classification"] = pd.to_numeric(df["classification"], errors="coerce")
df = df.dropna(subset=["classification"])

# ==============================
# 4. Replace '?' with NaN
# ==============================
df.replace("?", np.nan, inplace=True)

# ==============================
# 5. Convert Numeric Columns
# ==============================
numeric_cols = [
    "age","bp","sg","al","su","bgr","bu","sc",
    "sod","pot","hemo","pcv","wc","rc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==============================
# 6. Fill Missing Values
# ==============================
# Numeric → median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical → mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==============================
# 7. Encode Categorical Columns
# ==============================
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ==============================
# 8. Split Features & Target
# ==============================
X = df.drop(columns=["classification"])
y = df["classification"]

# ==============================
# 9. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 10. Train Random Forest
# ==============================
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# 11. Evaluate on Test Set
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("TEST SET RESULTS")
print("==============================")
print("CKD Model Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 12. Cross Validation (5-Fold)
# ==============================
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\n==============================")
print("CROSS VALIDATION RESULTS")
print("==============================")
print("CV Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", round(cv_scores.mean(), 4))

# ==============================
# 13. Feature Importance
# ==============================
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n==============================")
print("TOP 10 IMPORTANT FEATURES")
print("==============================")
print(importances.sort_values(ascending=False).head(10))

# ==============================
# 14. Save Model & Metadata
# ==============================
with open("models/ckd_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/ckd_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("models/ckd_label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("\nCKD model and metadata saved successfully.")
