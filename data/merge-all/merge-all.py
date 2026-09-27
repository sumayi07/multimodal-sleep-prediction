# combine_all_data.py
import pandas as pd

def combine_all_data(phone_sleep_file, mood_file, exercise_file, output_file):
    """
    Combines phone-sleep, pre-sleep mood, and exercise data, keeping only participants 
    that exist in ALL three datasets.
    
    Args:
        phone_sleep_file (str): Path to phone-sleep CSV
        mood_file (str): Path to pre-sleep mood CSV  
        exercise_file (str): Path to exercise CSV
        output_file (str): Path to save combined data
    """
    
    print("=" * 60)
    print("COMBINING PHONE-SLEEP, PRE-SLEEP MOOD, AND EXERCISE DATA")
    print("=" * 60)
    
    # Load all datasets
    phone_sleep_df = pd.read_csv(phone_sleep_file)
    mood_df = pd.read_csv(mood_file)
    exercise_df = pd.read_csv(exercise_file)
    
    print(f"\n📊 INITIAL DATASETS:")
    print(f"Phone-Sleep: {phone_sleep_df['participant_id'].nunique()} participants, {len(phone_sleep_df)} rows")
    print(f"Pre-Sleep Mood: {mood_df['participant_id'].nunique()} participants, {len(mood_df)} rows")
    print(f"Exercise: {exercise_df['participant_id'].nunique()} participants, {len(exercise_df)} rows")
    
    # Get unique participants from each dataset
    phone_sleep_participants = set(phone_sleep_df['participant_id'].unique())
    mood_participants = set(mood_df['participant_id'].unique())
    exercise_participants = set(exercise_df['participant_id'].unique())
    
    # Find participants that exist in ALL three datasets
    common_participants = phone_sleep_participants & mood_participants & exercise_participants
    
    print(f"\n👥 PARTICIPANT OVERLAP:")
    print(f"Participants in ALL three datasets: {len(common_participants)}")
    print(f"Participants only in Phone-Sleep: {len(phone_sleep_participants - common_participants)}")
    print(f"Participants only in Pre-Sleep Mood: {len(mood_participants - common_participants)}")
    print(f"Participants only in Exercise: {len(exercise_participants - common_participants)}")
    
    if not common_participants:
        print("❌ ERROR: No common participants found across all three datasets!")
        return None
    
    # Filter each dataset to only include common participants
    phone_sleep_filtered = phone_sleep_df[phone_sleep_df['participant_id'].isin(common_participants)].copy()
    mood_filtered = mood_df[mood_df['participant_id'].isin(common_participants)].copy()
    exercise_filtered = exercise_df[exercise_df['participant_id'].isin(common_participants)].copy()
    
    print(f"\n✅ FILTERED DATASETS (common participants only):")
    print(f"Phone-Sleep: {phone_sleep_filtered['participant_id'].nunique()} participants, {len(phone_sleep_filtered)} rows")
    print(f"Pre-Sleep Mood: {mood_filtered['participant_id'].nunique()} participants, {len(mood_filtered)} rows")
    print(f"Exercise: {exercise_filtered['participant_id'].nunique()} participants, {len(exercise_filtered)} rows")
    
    # Rename exercise columns as requested
    exercise_renamed = exercise_filtered.rename(columns={
        'Fat Burn_minutes': 'fatBurnMinutes',
        'Cardio_minutes': 'cardioMinutes', 
        'NumberSteps': 'numberSteps'
    })
    
    # Select and rename mood columns
    mood_renamed = mood_filtered.rename(columns={
        'pre_sleep_avg_stress': 'stress',
        'pre_sleep_avg_anxiety': 'anxiety'
    })
    
    # Drop unnecessary columns from mood data (keep only what we need)
    mood_columns_to_keep = ['participant_id', 'date', 'stress', 'anxiety']
    mood_final = mood_renamed[mood_columns_to_keep]
    
    print(f"\n🔄 MERGING DATASETS...")
    
    # First merge: Phone-Sleep + Pre-Sleep Mood
    merged_1 = pd.merge(
        phone_sleep_filtered,
        mood_final,
        on=['participant_id', 'date'],
        how='left',  # Keep all phone-sleep records, even if no mood data for that date
        suffixes=('', '_mood')
    )
    
    print(f"After Phone-Sleep + Mood merge: {len(merged_1)} rows")
    
    # Second merge: Add Exercise
    final_merged = pd.merge(
        merged_1,
        exercise_renamed,
        on=['participant_id', 'date'],
        how='left',  # Keep all records, even if no exercise data for that date
        suffixes=('', '_exercise')
    )
    
    print(f"After adding Exercise: {len(final_merged)} rows")
    
    # Clean up column names (remove any duplicates)
    final_merged = final_merged.loc[:, ~final_merged.columns.duplicated()]
    
    # Select and reorder final columns as specified
    final_columns = [
        'participant_id', 'date', 'minutesAsleep', 'efficiency', 'restfulness',
        'lastSessionDuration', 'phoneSleepGap', 'dayPhoneDuration',
        'stress', 'anxiety', 'fatBurnMinutes', 'cardioMinutes', 'numberSteps'
    ]
    
    # Only include columns that actually exist in the final dataset
    available_columns = [col for col in final_columns if col in final_merged.columns]
    final_merged = final_merged[available_columns]
    
    # Report on missing data in final dataset
    print(f"\n📈 FINAL DATASET STATISTICS:")
    print(f"Total participants: {final_merged['participant_id'].nunique()}")
    print(f"Total rows: {len(final_merged)}")
    print(f"Rows with stress data: {final_merged['stress'].notna().sum()} ({final_merged['stress'].notna().mean():.1%})")
    print(f"Rows with anxiety data: {final_merged['anxiety'].notna().sum()} ({final_merged['anxiety'].notna().mean():.1%})")
    print(f"Rows with exercise data: {final_merged['fatBurnMinutes'].notna().sum()} ({final_merged['fatBurnMinutes'].notna().mean():.1%})")
    print(f"Rows with all data (stress + anxiety + exercise): {final_merged[['stress', 'anxiety', 'fatBurnMinutes']].notna().all(axis=1).sum()}")
    
    # Save final dataset
    final_merged.to_csv(output_file, index=False)
    print(f"\n💾 Saved combined data to: {output_file}")
    
    # Show sample of final data
    print(f"\n📋 SAMPLE OF COMBINED DATA:")
    print(final_merged.head(10))
    
    return final_merged

# Alternative version with inner join (only dates with all three data points)
def combine_all_data_inner_join(phone_sleep_file, mood_file, exercise_file, output_file):
    """
    Combines data using INNER JOIN - only keeps dates where all three datasets have data
    """
    
    # Load datasets
    phone_sleep_df = pd.read_csv(phone_sleep_file)
    mood_df = pd.read_csv(mood_file)
    exercise_df = pd.read_csv(exercise_file)
    
    print("USING INNER JOIN (only complete cases)")
    
    # Find common participants
    common_participants = set(phone_sleep_df['participant_id'].unique()) & \
                        set(mood_df['participant_id'].unique()) & \
                        set(exercise_df['participant_id'].unique())
    
    # Filter to common participants
    phone_sleep_filtered = phone_sleep_df[phone_sleep_df['participant_id'].isin(common_participants)]
    mood_filtered = mood_df[mood_df['participant_id'].isin(common_participants)]
    exercise_filtered = exercise_df[exercise_df['participant_id'].isin(common_participants)]
    
    # Rename columns
    mood_renamed = mood_filtered.rename(columns={
        'pre_sleep_avg_stress': 'stress',
        'pre_sleep_avg_anxiety': 'anxiety'
    })
    mood_final = mood_renamed[['participant_id', 'date', 'stress', 'anxiety']]
    
    exercise_renamed = exercise_filtered.rename(columns={
        'Fat Burn_minutes': 'fatBurnMinutes',
        'Cardio_minutes': 'cardioMinutes',
        'NumberSteps': 'numberSteps'
    })
    
    # Merge with inner join (only dates with all three)
    merged_1 = pd.merge(phone_sleep_filtered, mood_final, on=['participant_id', 'date'], how='inner')
    final_merged = pd.merge(merged_1, exercise_renamed, on=['participant_id', 'date'], how='inner')
    
    # Select final columns
    final_columns = [
        'participant_id', 'date', 'minutesAsleep', 'efficiency', 'restfulness',
        'lastSessionDuration', 'phoneSleepGap', 'dayPhoneDuration',
        'stress', 'anxiety', 'fatBurnMinutes', 'cardioMinutes', 'numberSteps'
    ]
    available_columns = [col for col in final_columns if col in final_merged.columns]
    final_merged = final_merged[available_columns]
    
    print(f"Inner join result: {final_merged['participant_id'].nunique()} participants, {len(final_merged)} rows")
    
    final_merged.to_csv(output_file, index=False)
    return final_merged

# Example usage
if __name__ == "__main__":
    # Main version (recommended - keeps all sleep dates)
    combined_data = combine_all_data(
        phone_sleep_file="filtered_phone_sleep_data.csv",  # Your phone-sleep data
        mood_file="pre_sleep_mood.csv",                   # Your pre-sleep mood data
        exercise_file="exercise.csv",                     # Your exercise data
        output_file="final_combined_data.csv"
    )
    
    # # Alternative: Inner join version (only complete cases)
    # combined_data = combine_all_data_inner_join(
    #     phone_sleep_file="filtered_phone_sleep_data.csv",
    #     mood_file="pre_sleep_mood.csv", 
    #     exercise_file="exercise.csv",
    #     output_file="final_combined_data_inner.csv"
    # )