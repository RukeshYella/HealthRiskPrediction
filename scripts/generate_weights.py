# scripts/generate_weights.py
import pandas as pd
import numpy as np

# Load your manually sampled 60-case file
df = pd.read_csv("data/sample_60_cases.csv")

# Correlate all features with Pain severity
correlations = df.corr(numeric_only=True)['Pain severity'].drop('Pain severity')
top_features = correlations.abs().sort_values(ascending=False).head(10)

# Normalize weights
weights = (top_features / top_features.sum()).round(3)

# Combine into a DataFrame
weight_table = pd.DataFrame({
    "Feature": top_features.index,
    "Correlation": correlations[top_features.index].round(3),
    "Assigned Weight": weights.values
})

# Build score formula
formula_parts = [f"{w} × {feat}" for feat, w in zip(weight_table['Feature'], weight_table['Assigned Weight'])]
score_formula = " + ".join(formula_parts)

# Save to view or edit manually later
weight_table.to_csv("data/scoring_weights.csv", index=False)

print("✅ Top weighted features extracted.")
print("📁 Saved to: data/scoring_weights.csv")
print("\n📐 Suggested Scoring Formula:")
print(f"score = {score_formula}")
