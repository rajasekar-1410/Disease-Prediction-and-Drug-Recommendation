from disease_prediction.predict_disease import predict_disease

# Choose some symptoms that EXIST in Training.csv
sample_symptoms = [
    "fever",
    "headache",
    "fatigue"
]

result = predict_disease(sample_symptoms)
print("Predicted Disease:", result)
