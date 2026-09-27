# merge_phone_sleep.py
import pandas as pd
import numpy as np
from datetime import timedelta

SLEEP_CSV = "sleep_master.csv"
PHONE_CSV = "phone.csv"
OUTPUT_CSV = "merged_phone_sleep.csv"

def main(sleep_csv=SLEEP_CSV, phone_csv=PHONE_CSV, out_csv=OUTPUT_CSV):
    # Load data
    sleep_df = pd.read_csv(sleep_csv)
    phone_df = pd.read_csv(phone_csv)
    
    print("Initial data shapes:")
    print(f"Sleep: {sleep_df.shape}, Phone: {phone_df.shape}")
    
    # Convert datetime columns
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'])
    
    phone_df['session_start'] = pd.to_datetime(phone_df['session_start'])
    phone_df['session_stop'] = pd.to_datetime(phone_df['session_stop'])
    
    # Sort both datasets by time
    sleep_df = sleep_df.sort_values(['participant_id', 'startTime'])
    phone_df = phone_df.sort_values(['participant_id', 'session_stop'])
    
    # Calculate restfulness
    sleep_df['restfulness'] = ((sleep_df['remDuration'] + sleep_df['deepDuration']) / 
                              sleep_df['timeInBed'] * 100).round(2)

    # Convert to NaN where both rem and deep are 0 (no stage data recorded)
    no_stage_data = (sleep_df['remDuration'] == 0) & (sleep_df['deepDuration'] == 0)
    sleep_df.loc[no_stage_data, 'restfulness'] = np.nan
    
    # For each participant, process sleep sessions in chronological order
    results = []
    
    for participant_id in sleep_df['participant_id'].unique():
        participant_sleep = sleep_df[sleep_df['participant_id'] == participant_id].copy()
        participant_phone = phone_df[phone_df['participant_id'] == participant_id].copy()
        
        # Assign sleep sessions to dates based on START time
        participant_sleep['assigned_date'] = participant_sleep['startTime'].dt.date
        
        # Group by assigned date and keep the EARLIEST sleep session for each date
        daily_sleep = participant_sleep.loc[participant_sleep.groupby('assigned_date')['startTime'].idxmin()]
        
        # Process sleep sessions in chronological order
        daily_sleep = daily_sleep.sort_values('startTime')
        
        for i, (idx, sleep_row) in enumerate(daily_sleep.iterrows()):
            sleep_start = sleep_row['startTime']
            sleep_end = sleep_row['endTime']
            assigned_date = sleep_row['assigned_date']
            
            # Determine the start of the time window for phone usage
            if i == 0:
                # First sleep session: use all phone sessions before this sleep
                window_start = None  # No lower bound
            else:
                # Subsequent sleep sessions: use phone sessions after previous sleep ended
                prev_sleep_end = daily_sleep.iloc[i-1]['endTime']
                window_start = prev_sleep_end
            
            # Find phone sessions in the appropriate time window
            if window_start is None:
                # First sleep: all sessions before this sleep
                pre_sleep_sessions = participant_phone[
                    participant_phone['session_stop'] < sleep_start
                ]
            else:
                # Subsequent sleeps: sessions after previous sleep ended and before current sleep started
                pre_sleep_sessions = participant_phone[
                    (participant_phone['session_stop'] > window_start) & 
                    (participant_phone['session_stop'] < sleep_start)
                ]
            
            # Calculate lastSessionDuration and phoneSleepGap
            if not pre_sleep_sessions.empty:
                last_session = pre_sleep_sessions.loc[pre_sleep_sessions['session_stop'].idxmax()]
                last_session_duration = last_session['session_duration']
                phone_sleep_gap = (sleep_start - last_session['session_stop']).total_seconds() / 60.0
            else:
                last_session_duration = np.nan
                phone_sleep_gap = np.nan
            
            # Calculate dayPhoneDuration: total of all sessions in the window
            day_phone_duration = pre_sleep_sessions['session_duration'].sum()
            
            # Create result row
            result_row = {
                'participant_id': participant_id,
                'date': assigned_date,
                'minutesAsleep': sleep_row['minutesAsleep'],
                'efficiency': sleep_row['efficiency'],
                'restfulness': sleep_row['restfulness'],
                'lastSessionDuration': last_session_duration,
                'phoneSleepGap': phone_sleep_gap,
                'dayPhoneDuration': day_phone_duration
            }
            
            results.append(result_row)
    
    # Create final dataframe
    final_df = pd.DataFrame(results)
    
    # Print diagnostics
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Missingness rates:")
    print(f"lastSessionDuration: {final_df['lastSessionDuration'].isna().mean():.1%}")
    print(f"phoneSleepGap: {final_df['phoneSleepGap'].isna().mean():.1%}")
    print(f"dayPhoneDuration: {final_df['dayPhoneDuration'].isna().mean():.1%}")
    
    # Save results
    final_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved to: {out_csv}")
    
    return final_df

if __name__ == "__main__":
    df = main()
    
    # Show sample of results
    print("\nSample of merged data:")
    print(df.head(10))