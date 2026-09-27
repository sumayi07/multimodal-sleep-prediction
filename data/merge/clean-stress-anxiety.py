import pandas as pd
from datetime import timedelta

def process_ema_mood_for_sleep(
    anxiety_file, 
    stress_file, 
    sleep_file, 
    output_file, 
    include_debug=False, 
    print_summary=True
):
    """
    Calculates average anxiety and stress scores for the period BEFORE each sleep session.
    Only counts surveys that occurred after the previous sleep ended, with a 24-hour maximum window.
    """
    # Load and clean EMA data
    anxiety_df = pd.read_csv(anxiety_file)
    anxiety_df = anxiety_df[anxiety_df['Finished'] == 1]
    anxiety_df['datetime'] = pd.to_datetime(anxiety_df['start_ts'], utc=True).dt.tz_convert(None)
    
    stress_df = pd.read_csv(stress_file)
    stress_df = stress_df[stress_df['Finished'] == 1]
    stress_df['datetime'] = pd.to_datetime(stress_df['start_ts'], utc=True).dt.tz_convert(None)
    
    # Load and process sleep data
    sleep_df = pd.read_csv(sleep_file)
    sleep_df['sleep_start'] = pd.to_datetime(sleep_df['startTime'])
    sleep_df['sleep_end'] = pd.to_datetime(sleep_df['endTime'])
    sleep_df['sleep_date'] = pd.to_datetime(sleep_df['date']).dt.date
    
    # Sort sleep data chronologically by participant
    sleep_df = sleep_df.sort_values(['participant_id', 'sleep_start'])
    
    results = []
    
    # Process by participant
    for participant_id in sleep_df['participant_id'].unique():
        participant_sleep = sleep_df[sleep_df['participant_id'] == participant_id]
        participant_anxiety = anxiety_df[anxiety_df['participant_id'] == participant_id]
        participant_stress = stress_df[stress_df['participant_id'] == participant_id]
        
        last_sleep_end = None
        
        for _, sleep_row in participant_sleep.iterrows():
            sleep_start = sleep_row['sleep_start']
            sleep_end = sleep_row['sleep_end']
            sleep_date = sleep_row['sleep_date']
            
            twenty_four_hours_before = sleep_start - timedelta(hours=24)
            
            # Determine window start time
            if last_sleep_end is None:
                window_start = twenty_four_hours_before
            else:
                window_start = max(last_sleep_end, twenty_four_hours_before)
            
            # Filter surveys within the valid pre-sleep window
            pre_sleep_anxiety = participant_anxiety[
                (participant_anxiety['datetime'] >= window_start) & 
                (participant_anxiety['datetime'] < sleep_start)
            ]
            pre_sleep_stress = participant_stress[
                (participant_stress['datetime'] >= window_start) & 
                (participant_stress['datetime'] < sleep_start)
            ]
            
            # Calculate metrics
            row_data = {
                'participant_id': participant_id,
                'date': sleep_date,
                'sleep_start': sleep_start,
                'pre_sleep_avg_anxiety': pre_sleep_anxiety['anxiety'].mean() if not pre_sleep_anxiety.empty else None,
                'anxiety_surveys_count': len(pre_sleep_anxiety),
                'pre_sleep_avg_stress': pre_sleep_stress['stressd'].mean() if not pre_sleep_stress.empty else None,
                'stress_surveys_count': len(pre_sleep_stress)
            }
            
            # Add debug columns if requested
            if include_debug:
                row_data['sleep_end'] = sleep_end
                row_data['window_start_used'] = window_start
                
            results.append(row_data)
            last_sleep_end = sleep_end
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False)
    print(f"Pre-sleep mood data saved to {output_file}")
    
    if print_summary:
        print(f"\n📊 PRE-SLEEP MOOD SUMMARY:")
        print(f"Total sleep sessions: {len(final_df)}")
        print(f"Sessions with anxiety data: {final_df['pre_sleep_avg_anxiety'].notna().sum()} ({final_df['pre_sleep_avg_anxiety'].notna().mean():.1%})")
        print(f"Sessions with stress data: {final_df['pre_sleep_avg_stress'].notna().sum()} ({final_df['pre_sleep_avg_stress'].notna().mean():.1%})")
        print(f"Sessions with both mood measures: {final_df[['pre_sleep_avg_anxiety', 'pre_sleep_avg_stress']].notna().all(axis=1).sum()}")
    
    return final_df