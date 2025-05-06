# scripts/sample_cases.py
import pandas as pd
import os

# Load your latest processed dataset
df = pd.read_csv("data/processed_data.csv")

# Make sure target column is present
assert 'Pain severity' in df.columns, "❌ 'Pain severity' column not found!"

# Grouped sampling: get up to 5 records from each severity level (0 to 10)
sample_df = pd.DataFrame()

for severity in range(11):
    group = df[df['Pain severity'] == severity]
    if not group.empty:
        count = min(5, len(group))
        sample_df = pd.concat([sample_df, group.sample(n=count, random_state=42)])

# Shuffle the combined sample
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the sample to a CSV file
output_path = "data/sample_60_cases.csv"
sample_df.to_csv(output_path, index=False)

print("✅ Sampled 60 cases saved successfully.")
print(f"📁 File saved at: {output_path}")
print(f"🔢 Rows: {sample_df.shape[0]} (Target levels: {sample_df['Pain severity'].nunique()} present)")
