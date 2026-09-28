import pandas as pd

# Read combined RMSE data
df = pd.read_csv("rmse_data.csv")

# Separate baseline and Random Forest models
baselines = df[df['Model Type'] == 'Baseline']
models = df[df['Model Type'] == 'Random Forest']

results = []
for _, row in models.iterrows():
    baseline_val = baselines[baselines['Target'] == row['Target']].iloc[0]
    improvement = 100 * (baseline_val['RMSE (Mean)'] - row['RMSE (Mean)']) / baseline_val['RMSE (Mean)']
    
    results.append({
        'Target': row['Target'],
        'Predictor Group': row['Predictor Group'],
        'Percentage Improvement': improvement,
        'Original RMSE': row['RMSE (Mean)'],
        'Baseline RMSE': baseline_val['RMSE (Mean)']
    })

# Save improvement results
improvement_df = pd.DataFrame(results)
improvement_df.to_csv("improvement_data.csv", index=False)
print("Saved percentage improvements to 'improvement_data.csv'")