# scripts/clean_data.py
import pandas as pd
import os

# Load original dataset
data_path = os.path.join("data", "Health_observations.csv")
df = pd.read_csv(data_path)

# Strip columns
df.columns = df.columns.str.strip()

# Rename target column
df.rename(columns={
    'Pain severity - 0-10 verbal numeric rating [Score] - Reported': 'Pain severity'
}, inplace=True)

# Standardize smoking status
df['Tobacco smoking status'] = df['Tobacco smoking status'].replace({
    'Never smoked tobacco (finding)': 'NO',
    'Ex-smoker (finding)': 'EX',
    'Smoker (finding)': 'YES'
})

# Save cleaned dataset
df.to_csv(data_path, index=False)

print(" Data cleaned and saved to 'Health_observations.csv'")
