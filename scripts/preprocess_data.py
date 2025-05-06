# scripts/preprocess_data.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

# Load original data
data_path = os.path.join("data", "Health_observations.csv")
df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# --- Feature Engineering ---
df['BMI'] = df['Body Weight'] / ((df['Body Height'] / 100) ** 2)

# Smoking status mapping (based on actual unique values)
smoking_map = {
    'NO': 'NO',
    'EX': 'EX',
    'Smokes tobacco daily (finding)': 'YES'
}
df['Tobacco smoking status'] = df['Tobacco smoking status'].map(smoking_map)

# Encode mapped smoking status
encode_map = {'NO': 0, 'EX': 1, 'YES': 2}
df['Smoking_encoded'] = df['Tobacco smoking status'].map(encode_map)

# Drop original column
df.drop(columns=['Tobacco smoking status'], inplace=True)

# Drop rows with any missing values after mapping
df.dropna(inplace=True)

# Scale numerical columns
numerical_cols = [
    'Body Height',
    'Body Weight',
    'Diastolic Blood Pressure',
    'Heart rate',
    'Respiratory rate',
    'Systolic Blood Pressure',
    'BMI'
]

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Move target column to end
target_col = "Pain severity"
cols = [col for col in df.columns if col != target_col] + [target_col]
df = df[cols]

# Save processed data and scaler
processed_path = os.path.join("data", "processed_data.csv")
df.to_csv(processed_path, index=False)
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Feature engineering and scaling complete.")
print(f"📁 Processed dataset saved to {processed_path}")
print("📦 Scaler saved to models/scaler.pkl")
