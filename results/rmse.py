import pandas as pd

# Load Random Forest and Baseline results
rf_df = pd.read_csv("results.csv")
baseline_df = pd.read_csv("rest_baseline_final.csv")

# Standardize model types and predictor group labels
rf_df["Model Type"] = "Random Forest"
baseline_df["Model Type"] = "Baseline"
baseline_df["Predictor Group"] = "baseline"

# Concatenate datasets
combined_df = pd.concat([rf_df, baseline_df], ignore_index=True)

# Select and reorder final columns
combined_df = combined_df[[
    "Target", 
    "Model Type", 
    "Predictor Group", 
    "RMSE (Mean)", 
    "RMSE (Std)"
]]

# Save master dataset
combined_df.to_csv("rmse_data.csv", index=False)
print("Saved combined results to 'rmse_data.csv'")