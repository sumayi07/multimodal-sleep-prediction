import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from datetime import datetime

def print_progress(message):
    """Helper function to print timestamped progress messages"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def main():
    # Load data
    print_progress("Loading data...")
    data = pd.read_csv("final_combined_data.csv")  # ← Use your correct file!

    # --- Data Preprocessing ---
    print_progress("Preprocessing data...")

    # 1. Filter rows where minutesAsleep = 0
    data = data[data['minutesAsleep'] != 0]
    data = data[data['dayPhoneDuration'].notna()]

    # 2. Add missing indicators for stress and anxiety
    data["stress_missing"] = data["stress"].isna().astype(int)
    data["anxiety_missing"] = data["anxiety"].isna().astype(int)

    # 3. Quick scale check (no artificial scaling)
    print("\n=== DATA SCALE CHECK ===")
    for target in ["efficiency", "minutesAsleep", "restfulness"]:
        if target in data.columns:
            target_data = data[target].dropna()
            print(f"{target}: Min: {target_data.min()}, Max: {target_data.max()}, Mean: {target_data.mean():.2f}")

    # Define sleep targets
    sleep_targets = ["efficiency", "minutesAsleep", "restfulness"]

    # Setup 10-fold cross validation grouped by participant
    group_kfold = GroupKFold(n_splits=10)
    participant_ids = data["participant_id"]

    # Store results
    baseline_results = []

    for target in sleep_targets:
        print_progress(f"\nProcessing target: {target}")
        
        # Skip rows where target is missing
        valid_rows = data[target].notna()
        y = data.loc[valid_rows, target]
        groups = participant_ids[valid_rows]
        
        if len(y) == 0:
            print_progress(f"Skipping {target} - no valid data")
            continue
            
        rmse_scores = []
        sample_counts = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(
            group_kfold.split(y, y, groups=groups), 1):
            
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Baseline model: predict mean of training set
            y_pred = [y_train.mean()] * len(y_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
            sample_counts.append(len(test_idx))
            
            if fold_idx % 2 == 0 or fold_idx == 10:
                print_progress(f"  Fold {fold_idx}/10 (Current Avg RMSE: {np.mean(rmse_scores):.2f})")
        
        baseline_results.append({
            "Target": target,
            "RMSE (Mean)": np.mean(rmse_scores),
            "RMSE (Std)": np.std(rmse_scores),
            "Avg Test Samples per Fold": np.mean(sample_counts),
            "Total Samples": len(y)
        })

    # Save results
    baseline_df = pd.DataFrame(baseline_results)
    print("\n=== CROSS-VALIDATED BASELINE RESULTS ===")
    print(baseline_df)

    # Compare to standard deviation
    std_comparison = data[sleep_targets].std().to_frame("std")
    std_comparison["RMSE"] = baseline_df.set_index("Target")["RMSE (Mean)"]
    std_comparison["Ratio (RMSE/std)"] = std_comparison["RMSE"] / std_comparison["std"]
    print("\n=== RMSE vs STANDARD DEVIATION ===")
    print(std_comparison)

    baseline_df.to_csv("rest_baseline_final.csv", index=False)
    std_comparison.to_csv("rest_rmse_std_final.csv", index=True)
    print_progress("Results saved to 'rest_baseline_final.csv' and 'rest_rmse_std_final.csv'")

if __name__ == "__main__":
    main()