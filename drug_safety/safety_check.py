import pandas as pd
from itertools import combinations
import os

def check_drug_safety(drug_list):
    """
    drug_list: list of drug names selected by user
    returns: list of safety messages (strings only)
    """

    safety_messages = []

    # Defensive check
    if not isinstance(drug_list, list) or len(drug_list) < 2:
        return ["⚠️ Please select at least two drugs for interaction check"]

    # Load interaction data safely
    file_path = os.path.join("drug_safety", "interaction_data.csv")

    try:
        interaction_df = pd.read_csv(file_path)
    except Exception as e:
        return ["⚠️ Drug interaction data could not be loaded"]

    unsafe_found = False

    # Generate all possible drug pairs
    for drug1, drug2 in combinations(drug_list, 2):
        match = interaction_df[
            (
                (interaction_df["drug1"] == drug1) &
                (interaction_df["drug2"] == drug2)
            ) |
            (
                (interaction_df["drug1"] == drug2) &
                (interaction_df["drug2"] == drug1)
            )
        ]

        if not match.empty:
            unsafe_found = True
            interaction_info = str(match.iloc[0]["interaction"])
            message = f"⚠️ Interaction between {drug1} and {drug2}: {interaction_info}"
            safety_messages.append(message)

    if not unsafe_found:
        safety_messages.append("✅ No known drug–drug interactions detected")

    return safety_messages
