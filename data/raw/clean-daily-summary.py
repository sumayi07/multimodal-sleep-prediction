import os
import pandas as pd
from config import fitbit_data_folder

def combine_fitbit_daily_summaries(root_folder, output_file):

    columns_to_keep = ['Fat Burn_minutes', 'Cardio_minutes', 'NumberSteps', 'Timestamp']

    all_data = []
    processed_files = 0
    missing_columns = set()
    
    for filename in os.listdir(root_folder):
        if filename.endswith('.csv'):
            try:
                file_path = os.path.join(root_folder, filename)
                df = pd.read_csv(file_path)
                participant_id = filename[:-4]

                missing_cols = [col for col in columns_to_keep if col not in df.columns]
                if missing_cols:
                    missing_columns.update(missing_cols)
                    continue

                df = df[columns_to_keep].copy()

                # rename timestamp → date
                df['date'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.date
                df = df.drop(columns=['Timestamp'])

                df['participant_id'] = participant_id
                all_data.append(df)
                processed_files += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    if not all_data:
        print("No valid data found!")
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)

    combined_df = combined_df[['participant_id','date','Fat Burn_minutes','Cardio_minutes','NumberSteps']]
    
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully processed {processed_files} files")
    print(f"Saved combined data to {output_file}")
    print(f"Total participants: {combined_df['participant_id'].nunique()}")
    print(f"Total records: {len(combined_df)}")
    if missing_columns:
        print(f"Warning: some files skipped due to missing columns: {missing_columns}")

if __name__ == "__main__":
    output_csv = "exercise.csv"
    combine_fitbit_daily_summaries(fitbit_data_folder, output_csv)