# scripts/enhance_features.py
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Load original data
data_path = os.path.join("data", "Health_observations.csv")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# --- Basic Feature Engineering ---
df['BMI'] = df['Body Weight'] / ((df['Body Height'] / 100) ** 2)
df['Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
df['BP_Ratio'] = df['Systolic Blood Pressure'] / df['Diastolic Blood Pressure']
df['HRxRR'] = df['Heart rate'] * df['Respiratory rate']

# --- Normalize and Encode Smoking Status ---
smoking_map = {
    'NO': 'NO',
    'EX': 'EX',
    'Smokes tobacco daily (finding)': 'YES'
}
df['Tobacco smoking status'] = df['Tobacco smoking status'].map(smoking_map)
smoking_encode = {'NO': 0, 'EX': 1, 'YES': 2}
df['Smoking_encoded'] = df['Tobacco smoking status'].map(smoking_encode)

df.drop(columns=['Tobacco smoking status'], inplace=True)
df.dropna(inplace=True)

# --- Advanced Feature Engineering ---
df['HeartStress'] = df['Heart rate'] / df['BMI']
df['SmokingBMI'] = df['Smoking_encoded'] * df['BMI']
df['Pulse_HR'] = df['Pulse_Pressure'] * df['Heart rate']
df['BP_Smoking'] = df['BP_Ratio'] * df['Smoking_encoded']

# --- BMI Category Binning ---
df['BMI_Category'] = pd.cut(
    df['BMI'],
    bins=[-float("inf"), 18.5, 24.9, 29.9, float("inf")],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# One-hot encode BMI category
df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

# --- Scaling ---
numerical_cols = [
    'Body Height', 'Body Weight',
    'Diastolic Blood Pressure', 'Systolic Blood Pressure',
    'Heart rate', 'Respiratory rate', 'BMI',
    'Pulse_Pressure', 'BP_Ratio', 'HRxRR',
    'HeartStress', 'SmokingBMI', 'Pulse_HR', 'BP_Smoking'
]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Reorder target to last
target_col = "Pain severity"
cols = [col for col in df.columns if col != target_col] + [target_col]
df = df[cols]

# Save dataset and scaler
processed_path = "data/processed_data.csv"
df.to_csv(processed_path, index=False)
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Advanced feature engineering complete.")
print(f"📁 Processed dataset saved to {processed_path}")
