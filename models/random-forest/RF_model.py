import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from datetime import datetime
import sys
from itertools import combinations

def print_progress(message):
    """Helper function to print timestamped progress messages"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def calculate_feature_importance_percentage(feature_importances, feature_names):
    """Convert feature importances to percentages"""
    total_importance = sum(feature_importances)
    if total_importance == 0:
        return {name: 0 for name in feature_names}

    importance_percentages = {}
    for name, importance in zip(feature_names, feature_importances):
        importance_percentages[name] = (importance / total_importance) * 100
    return importance_percentages

def main():
    # Load data
    print_progress("Loading data...")
    data = pd.read_csv("final_combined_data.csv")

    # Define predictor groups
    groups = {
        "phone": ["dayPhoneDuration", "lastSessionDuration", "phoneSleepGap"],
        "exercise": ["fatBurnMinutes", "cardioMinutes", "numberSteps"],
        "mood": ["anxiety", "stress"]
    }

    # Define targets
    sleep_targets = ["efficiency", "minutesAsleep", "restfulness"]

    # --- Data Preprocessing ---
    print_progress("Preprocessing data...")

    # 1. Filter rows where minutesAsleep = 0
    data = data[data['minutesAsleep'] != 0]

    # 2. Filter rows with missing dayPhoneDuration
    print_progress("Filtering rows with missing dayPhoneDuration...")
    data = data[data['dayPhoneDuration'].notna()]

    # 3. Add missing indicators
    data["restfulness_missing"] = data["restfulness"].isna().astype(int)
    data["stress_missing"] = data["stress"].isna().astype(int)
    data["anxiety_missing"] = data["anxiety"].isna().astype(int)

    # Check available features
    available_groups = {}
    for group_name, features in groups.items():
        available_features = [f for f in features if f in data.columns and not data[f].isna().all()]
        if available_features:
            available_groups[group_name] = available_features

    # Apply median imputation
    print_progress("Applying median imputation...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    median_imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = median_imputer.fit_transform(data[numeric_cols])

    # Modeling setup
    group_kfold = GroupKFold(n_splits=10)
    participant_ids = data["participant_id"]
    rf_results = []
    feature_importance_results = []

    # Model counting
    single_group_models = len(available_groups)
    two_group_combos = len(list(combinations(available_groups.keys(), 2)))
    full_model = 1 if available_groups else 0
    models_per_target = single_group_models + two_group_combos + full_model
    total_models = len(sleep_targets) * models_per_target
    completed_models = 0

    print_progress(f"Starting modeling for {len(sleep_targets)} targets and {total_models} total models...")

    # Main modeling loop
    for target_idx, target in enumerate(sleep_targets, 1):
        # Skip rows where target is missing
        valid_rows = data[target].notna()
        y = data.loc[valid_rows, target]
        X_filtered = data.loc[valid_rows]
        participant_ids_filtered = participant_ids[valid_rows]
        
        if y.nunique() > 1:
            print_progress(f"\nProcessing target {target_idx}/{len(sleep_targets)}: {target}")
            
            # Individual group models
            for group_name, group_vars in available_groups.items():
                print_progress(f"  - Training model with {group_name} features...")
                X = X_filtered[group_vars]
                rmse_scores = []
                fold_feature_importances = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(
                    group_kfold.split(X, y, groups=participant_ids_filtered), 1):
                    
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    
                    # Store feature importances for this fold
                    fold_feature_importances.append(model.feature_importances_)
                    
                    if fold_idx % 2 == 0 or fold_idx == 10:
                        print_progress(f"    Completed fold {fold_idx}/10 (Current RMSE: {np.mean(rmse_scores):.2f})")
                
                final_rmse = np.mean(rmse_scores)
                
                # Calculate average feature importances across folds
                avg_feature_importances = np.mean(fold_feature_importances, axis=0)
                feature_percentages = calculate_feature_importance_percentage(avg_feature_importances, group_vars)
                
                # Store main results
                rf_results.append({
                    "Target": target,
                    "Predictor Group": group_name,
                    "RMSE (Mean)": final_rmse,
                    "RMSE (Std)": np.std(rmse_scores)
                })
                
                # Store feature importance results
                for feature, importance_pct in feature_percentages.items():
                    feature_importance_results.append({
                        "Target": target,
                        "Predictor Group": group_name,
                        "Feature": feature,
                        "Importance %": importance_pct,
                        "Model RMSE": final_rmse
                    })
                
                completed_models += 1
                print_progress(f"  - {group_name} model complete! Avg RMSE: {final_rmse:.2f}")
                print_progress(f"Overall progress: {completed_models}/{total_models} models ({completed_models/total_models:.1%})")

            # Two-group combinations
            for group1, group2 in combinations(available_groups.keys(), 2):
                combo_name = f"{group1}+{group2}"
                print_progress(f"  - Training combination model: {combo_name}...")
                combo_features = available_groups[group1] + available_groups[group2]
                X_combo = X_filtered[combo_features]
                rmse_scores = []
                fold_feature_importances = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(
                    group_kfold.split(X_combo, y, groups=participant_ids_filtered), 1):
                    
                    X_train, X_test = X_combo.iloc[train_idx], X_combo.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    
                    # Store feature importances for this fold
                    fold_feature_importances.append(model.feature_importances_)
                    
                    if fold_idx % 2 == 0 or fold_idx == 10:
                        print_progress(f"    Completed fold {fold_idx}/10 (Current RMSE: {np.mean(rmse_scores):.2f})")
                
                final_rmse = np.mean(rmse_scores)
                
                # Calculate average feature importances across folds
                avg_feature_importances = np.mean(fold_feature_importances, axis=0)
                feature_percentages = calculate_feature_importance_percentage(avg_feature_importances, combo_features)
                
                # Store main results
                rf_results.append({
                    "Target": target,
                    "Predictor Group": combo_name,
                    "RMSE (Mean)": final_rmse,
                    "RMSE (Std)": np.std(rmse_scores)
                })
                
                # Store feature importance results
                for feature, importance_pct in feature_percentages.items():
                    feature_importance_results.append({
                        "Target": target,
                        "Predictor Group": combo_name,
                        "Feature": feature,
                        "Importance %": importance_pct,
                        "Model RMSE": final_rmse
                    })
                
                completed_models += 1
                print_progress(f"  - {combo_name} model complete! Avg RMSE: {final_rmse:.2f}")
                print_progress(f"Overall progress: {completed_models}/{total_models} models ({completed_models/total_models:.1%})")

            # Full combined model
            if available_groups:
                print_progress("  - Training full combined model...")
                all_features = []
                for group_vars in available_groups.values():
                    all_features.extend(group_vars)
                X_full = X_filtered[all_features]
                rmse_scores_full = []
                fold_feature_importances = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(
                    group_kfold.split(X_full, y, groups=participant_ids_filtered), 1):
                    
                    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_leaf=5,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse_scores_full.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    
                    # Store feature importances for this fold
                    fold_feature_importances.append(model.feature_importances_)
                    
                    if fold_idx % 2 == 0 or fold_idx == 10:
                        print_progress(f"    Completed fold {fold_idx}/10 (Current RMSE: {np.mean(rmse_scores_full):.2f})")
                
                final_rmse = np.mean(rmse_scores_full)
                
                # Calculate average feature importances across folds
                avg_feature_importances = np.mean(fold_feature_importances, axis=0)
                feature_percentages = calculate_feature_importance_percentage(avg_feature_importances, all_features)
                
                # Store main results
                rf_results.append({
                    "Target": target,
                    "Predictor Group": "all_combined",
                    "RMSE (Mean)": final_rmse,
                    "RMSE (Std)": np.std(rmse_scores_full)
                })
                
                # Store feature importance results
                for feature, importance_pct in feature_percentages.items():
                    feature_importance_results.append({
                        "Target": target,
                        "Predictor Group": "all_combined",
                        "Feature": feature,
                        "Importance %": importance_pct,
                        "Model RMSE": final_rmse
                    })
                
                completed_models += 1
                print_progress(f"  - Full combined model complete! Avg RMSE: {final_rmse:.2f}")
                print_progress(f"Overall progress: {completed_models}/{total_models} models ({completed_models/total_models:.1%})")
        else:
            print_progress(f"Skipping target {target} (no variance in data)")
            total_models -= models_per_target

    # Save results
    print_progress("\nAll models complete! Saving results...")

    # Save main results
    rf_results_df = pd.DataFrame(rf_results)
    print("\n=== FINAL RESULTS ===")
    print(rf_results_df)
    rf_results_df.to_csv("results.csv", index=False)

    # Save feature importance results
    feature_importance_df = pd.DataFrame(feature_importance_results)
    print("\n=== FEATURE IMPORTANCE RESULTS ===")
    print(feature_importance_df.head(10))
    feature_importance_df.to_csv("RF_feature_importance_percentages.csv", index=False)

    print_progress("Done")

    # Verify total models run
    print(f"\nTotal models executed: {len(rf_results)}")
    print(f"Expected models: {total_models} (adjusted for skipped targets)" if total_models != len(sleep_targets)*models_per_target else "")

if __name__ == "__main__":
    main()