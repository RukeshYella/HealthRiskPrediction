# main.py
import pandas as pd
import joblib

# Load model, features, and scaler
model = joblib.load("models/pain_severity_model.pkl")
feature_columns = joblib.load("models/model_features.pkl")
scaler = joblib.load("models/scaler.pkl")

print("\n🩺 Enter values for new patient (raw inputs):")

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("⚠️ Please enter a valid number.")

# Get inputs
height = get_float("Body Height (in cm): ")
weight = get_float("Body Weight (in kg): ")
dbp = get_float("Diastolic Blood Pressure: ")
sbp = get_float("Systolic Blood Pressure: ")
hr = get_float("Heart Rate: ")
rr = get_float("Respiratory Rate: ")

# Smoking status input
while True:
    smoke_input = input("Tobacco Smoking Status (Enter: NO = Never, EX = Ex-Smoker, YES = Smoker): ").strip().upper()
    map_smoke = {"NO": 0, "EX": 1, "YES": 2}
    if smoke_input in map_smoke:
        smoke_code = map_smoke[smoke_input]
        print(f"ℹ️ Note: {smoke_input} encoded as {smoke_code}")
        break
    print("⚠️ Please enter one of: NO / EX / YES")

# --- Feature Engineering ---
bmi = weight / ((height / 100) ** 2)
pulse_pressure = sbp - dbp
bp_ratio = sbp / dbp
hrxrr = hr * rr
heart_stress = hr / bmi
smoking_bmi = smoke_code * bmi
pulse_hr = pulse_pressure * hr
bp_smoking = bp_ratio * smoke_code

# BMI category (one-hot)
bmi_cat = 'Normal'
if bmi < 18.5:
    bmi_cat = 'Underweight'
elif 25 <= bmi < 30:
    bmi_cat = 'Overweight'
elif bmi >= 30:
    bmi_cat = 'Obese'

# One-hot encoding BMI category
bmi_onehot = {
    "BMI_Category_Normal": 0,
    "BMI_Category_Overweight": 0,
    "BMI_Category_Obese": 0
}
if bmi_cat != "Underweight":  # we dropped "Underweight" as base
    bmi_onehot[f"BMI_Category_{bmi_cat}"] = 1

# Raw input dict
raw_input = {
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
    'BP_Smoking': bp_smoking,
    'Smoking_encoded': smoke_code,
    **bmi_onehot
}

input_df = pd.DataFrame([raw_input])

# Apply scaling
numerical_cols = [
    'Body Height', 'Body Weight',
    'Diastolic Blood Pressure', 'Systolic Blood Pressure',
    'Heart rate', 'Respiratory rate', 'BMI',
    'Pulse_Pressure', 'BP_Ratio', 'HRxRR',
    'HeartStress', 'SmokingBMI', 'Pulse_HR', 'BP_Smoking'
]
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Align with model
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Predict
prediction = model.predict(input_df)[0]

# Risk label
def classify_risk(score):
    if score <= 4:
        return "Low"
    elif score <= 7:
        return "Medium"
    else:
        return "High"

risk = classify_risk(prediction)

# Output
print(f"\n🔮 Predicted Pain Severity: {prediction:.2f}")
print(f"⚠️ Health Risk Level: {risk}")
