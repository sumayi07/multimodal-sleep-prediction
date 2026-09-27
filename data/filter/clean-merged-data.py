# remove_no_phone_participants.py
import pandas as pd
import numpy as np

def remove_participants_without_phone_data(input_file, output_file):
    """
    Removes participants who have no phone usage data (all dayPhoneDuration = 0 or NaN)
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save filtered data
    """
    
    # Load data
    df = pd.read_csv(input_file)
    
    print(f"Initial dataset:")
    print(f"Total participants: {df['participant_id'].nunique()}")
    print(f"Total rows: {len(df)}")
    
    # Check for participants with no phone data
    # Group by participant and check if ALL their dayPhoneDuration values are 0 or NaN
    phone_stats = df.groupby('participant_id')['dayPhoneDuration'].agg([
        'count',  # Total rows for participant
        'sum',    # Sum of dayPhoneDuration
        lambda x: x.isna().sum()  # Count of NaN values
    ]).rename(columns={'<lambda>': 'nan_count'})
    
    # Participants to remove: those with sum = 0 OR all values are NaN
    participants_to_remove = phone_stats[
        (phone_stats['sum'] == 0) | 
        (phone_stats['nan_count'] == phone_stats['count'])
    ].index.tolist()
    
    # Filter the dataframe
    filtered_df = df[~df['participant_id'].isin(participants_to_remove)].copy()
    
    # Print results
    print(f"\nFiltering results:")
    print(f"Removed {len(participants_to_remove)} participants with no phone data:")
    for participant in participants_to_remove:
        print(f"  - {participant}")
    
    print(f"\nFinal dataset:")
    print(f"Remaining participants: {filtered_df['participant_id'].nunique()}")
    print(f"Remaining rows: {len(filtered_df)}")
    print(f"Removed {len(df) - len(filtered_df)} total rows")
    
    # Save filtered data
    filtered_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved filtered data to: {output_file}")
    
    return filtered_df

# Alternative version with more detailed diagnostics
def remove_participants_without_phone_data_detailed(input_file, output_file):
    """
    More detailed version with comprehensive reporting
    """
    
    # Load data
    df = pd.read_csv(input_file)
    
    print("=" * 60)
    print("REMOVING PARTICIPANTS WITH NO PHONE DATA")
    print("=" * 60)
    
    initial_participants = df['participant_id'].nunique()
    initial_rows = len(df)
    
    print(f"\nInitial dataset:")
    print(f"Participants: {initial_participants}")
    print(f"Total rows: {initial_rows}")
    
    # Analyze phone data distribution
    phone_summary = df.groupby('participant_id')['dayPhoneDuration'].agg([
        'count',
        'sum',
        'mean',
        lambda x: x.isna().sum(),
        lambda x: (x == 0).sum()
    ]).rename(columns={
        '<lambda_0>': 'nan_count',
        '<lambda_1>': 'zero_count'
    })
    
    # Identify participants to remove
    no_phone_participants = phone_summary[
        (phone_summary['sum'] == 0) | 
        (phone_summary['nan_count'] == phone_summary['count'])
    ]
    
    participants_to_remove = no_phone_participants.index.tolist()
    
    # Filter data
    filtered_df = df[~df['participant_id'].isin(participants_to_remove)]
    
    # Print detailed report
    print(f"\n📊 PHONE DATA ANALYSIS:")
    print(f"Participants with NO phone data: {len(participants_to_remove)}")
    print(f"Participants WITH phone data: {initial_participants - len(participants_to_remove)}")
    
    if participants_to_remove:
        print(f"\n🗑️ REMOVED PARTICIPANTS (no phone data):")
        for participant in participants_to_remove:
            stats = phone_summary.loc[participant]
            print(f"  - {participant}: {stats['count']} sleep records, all zero/NaN")
    
    print(f"\n✅ FINAL DATASET:")
    print(f"Remaining participants: {filtered_df['participant_id'].nunique()}")
    print(f"Remaining rows: {len(filtered_df)}")
    print(f"Rows removed: {initial_rows - len(filtered_df)}")
    
    # Show sample of remaining data
    print(f"\n📋 SAMPLE OF REMAINING DATA:")
    print(filtered_df[['participant_id', 'date', 'dayPhoneDuration']].head(10))
    
    # Save results
    filtered_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    return filtered_df

# Example usage
if __name__ == "__main__":
    # Use the detailed version for better reporting
    filtered_data = remove_participants_without_phone_data_detailed(
        input_file="merged_phone_sleep.csv",  # Replace with your file
        output_file="filtered_phone_sleep_data.csv"
    )