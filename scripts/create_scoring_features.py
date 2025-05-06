import joblib
import os

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Define the exact feature list used in the rule-based scoring logic
features = [
    "Body Height", "Body Weight", "Diastolic Blood Pressure", "Systolic Blood Pressure",
    "Heart rate", "Respiratory rate", "BMI", "Pulse_Pressure", "BP_Ratio", "HRxRR",
    "HeartStress", "SmokingBMI", "Pulse_HR", "BP_Smoking", "Smoking_encoded", "BMI_Category_Overweight"
]

# Save the list to a joblib file
joblib.dump(features, "models/scoring_features.pkl")
print("✅ scoring_features.pkl created successfully.")
