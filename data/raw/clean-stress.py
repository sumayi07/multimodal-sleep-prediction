import pandas as pd

def process_ema_stress_data(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Filter for only completed surveys (Finished == 1)
    df = df[df['Finished'] == 1]
    
    # Convert timestamp to datetime (handling timezones properly)
    df['datetime'] = pd.to_datetime(df['start_ts'], utc=True)
    # Extract just the date part (without time or timezone)
    df['date'] = df['datetime'].dt.date
    
    # For each participant-date pair, keep only one record (the first one)
    df = df.sort_values('datetime').drop_duplicates(['participant_id', 'date'])
    
    # Select only the columns we need
    result = df[['participant_id', 'date', 'stressd']]
        
    # Group by participant ID (creates a DataFrameGroupBy object)
    grouped = result.groupby('participant_id')
    
    # Save to CSV - this will maintain grouping by participant_id
    # Each participant's data will appear together in the CSV
    result.sort_values(['participant_id', 'date']).to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    # If you want to return the grouped object for further processing:
    return grouped

# Example usage
input_filename = 'stressd.csv'  # Change this to your input file name
output_filename = 'stress_cleaned.csv'
grouped_data = process_ema_stress_data(input_filename, output_filename)