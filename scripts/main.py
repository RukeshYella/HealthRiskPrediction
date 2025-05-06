# main.py
import pandas as pd
import joblib

# Load scaler and features
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/classifier_features.pkl")  # Or pain_severity_model.pkl if same structure

print("\n🩺 Enter values for new patient (raw inputs):")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("⚠️ Please enter a valid number.")

# Inputs
height = get_float("Body Height (in cm): ")
weight = get_float("Body Weight (in kg): ")
dbp = get_float("Diastolic Blood Pressure: ")
sbp = get_float("Systolic Blood Pressure: ")
hr = get_float("Heart Rate: ")
rr = get_float("Respiratory Rate: ")

while True:
    smoke = input("Tobacco Smoking Status (Enter: NO = Never, EX = Ex-Smoker, YES = Smoker): ").strip().upper()
    smoke_map = {"NO": 0, "EX": 1, "YES": 2}
    if smoke in smoke_map:
        smoke_code = smoke_map[smoke]
        break
    print("⚠️ Please enter one of: NO / EX / YES")

# Feature Engineering
bmi = weight / ((height / 100) ** 2)
pulse_pressure = sbp - dbp
bp_ratio = sbp / dbp
hrxrr = hr * rr
heart_stress = hr / bmi
smoking_bmi = smoke_code * bmi
pulse_hr = pulse_pressure * hr
bp_smoking = bp_ratio * smoke_code

# BMI Category
bmi_cat = "Normal"
if bmi < 18.5:
    bmi_cat = "Underweight"
elif bmi < 30 and bmi >= 25:
    bmi_cat = "Overweight"
elif bmi >= 30:
    bmi_cat = "Obese"

bmi_cat_overweight = 1 if bmi_cat == "Overweight" else 0

# Scaling
numerical = {
    'Body Height': height,
    'Body Weight': weight,
    'Diastolic Blood Pressure': dbp,
    'Systolic Blood Pressure': sbp,
    'Heart rate': hr,
    'Respiratory rate': rr,
    'BMI': bmi,
    'Pulse_Pressure': pulse_pressure,
    'BP_Ratio': bp_ratio,
    'HRxRR': hrxrr,
    'HeartStress': heart_stress,
    'SmokingBMI': smoking_bmi,
    'Pulse_HR': pulse_hr,
    'BP_Smoking': bp_smoking
}
scaled_df = pd.DataFrame([numerical])
scaled_array = scaler.transform(scaled_df)
scaled_values = dict(zip(numerical.keys(), scaled_array[0]))

# Manually add unscaled categorical flags
scaled_values['Smoking_encoded'] = smoke_code
scaled_values['BMI_Category_Overweight'] = bmi_cat_overweight

# Score Formula (from your earlier analysis)
score = (
    0.141 * scaled_values['SmokingBMI'] +
    0.138 * scaled_values['Smoking_encoded'] +
    0.112 * scaled_values['BP_Smoking'] +
    0.110 * scaled_values['Body Height'] +
    0.099 * scaled_values['Body Weight'] +
    0.084 * scaled_values['BMI'] +
    0.083 * scaled_values['Systolic Blood Pressure'] +
    0.079 * scaled_values['HeartStress'] +
    0.077 * scaled_values['Respiratory rate'] +
    0.077 * scaled_values['BMI_Category_Overweight']
)

# Classify Risk Level
if score <= 2.5:
    risk = "Low"
elif score <= 4.0:
    risk = "Medium"
else:
    risk = "High"

# Output
print(f"\n🧮 Calculated Score: {score:.2f}")
print(f"⚠️ Health Risk Level: {risk}")
