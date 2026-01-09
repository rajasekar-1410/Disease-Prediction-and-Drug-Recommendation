import pandas as pd
import os

def recommend_drugs(disease):
    """
    disease: predicted disease name (string)
    returns: list of recommended drug names (strings only)
    """

    # Defensive check
    if not isinstance(disease, str) or disease.strip() == "":
        return ["⚠️ Invalid disease input"]

    file_path = os.path.join("drug_recommendation", "drug_data.csv")

    try:
        drug_df = pd.read_csv(file_path)
    except Exception:
        return ["⚠️ Drug recommendation data could not be loaded"]

    # Check required columns
    if "disease" not in drug_df.columns or "drug" not in drug_df.columns:
        return ["⚠️ Drug dataset format is invalid"]

    drugs = drug_df[drug_df["disease"] == disease]["drug"].tolist()

    if not drugs:
        return ["Drug recommendation for this disease is outside the current scope"]

    # Ensure output is always list of strings
    return [str(drug) for drug in drugs]
