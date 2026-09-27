import os
import pandas as pd
from config import REALIZD_FOLDER

# ===== configurable =====
OUTPUT_CSV = "phone.csv"
MIN_SESSION_SECONDS = 60   # set to 30 if you prefer a 30s cutoff
# ========================

def process_phone_data(root_folder, output_file, min_session_seconds=60):
    all_sessions = []
    files_processed = 0
    total_rows = 0
    total_kept = 0

    for filename in os.listdir(root_folder):
        if not filename.endswith(".csv"):
            continue

        try:
            file_path = os.path.join(root_folder, filename)
            df = pd.read_csv(file_path)

            start_col = "session_start"
            stop_col  = "session_stop"
            pid_col   = "participant_id"

            df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
            df[stop_col]  = pd.to_datetime(df[stop_col], errors="coerce")

            df = df.dropna(subset=[start_col, stop_col])

            dur_seconds = (df[stop_col] - df[start_col]).dt.total_seconds()
            df["session_duration_seconds"] = dur_seconds
            df["session_duration"] = df["session_duration_seconds"] / 60.0

            before = len(df)
            df = df[df["session_duration_seconds"] >= min_session_seconds]
            after = len(df)

            total_rows += before
            total_kept += after

            df["date"] = df[start_col].dt.date

            all_sessions.append(df)
            files_processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not all_sessions:
        print("No valid data found! Check folder path and column names.")
        return

    all_sessions_df = pd.concat(all_sessions, ignore_index=True)

    # final output: session-level only (no more daily totals)
    final_output = all_sessions_df[[
        "participant_id",
        "date",
        "session_start",
        "session_stop",
        "session_duration"
    ]]

    final_output.to_csv(output_file, index=False)

    print(f"Processed files: {files_processed}")
    print(f"Rows before filtering: {total_rows}")
    print(f"Rows kept (>= {min_session_seconds}s): {total_kept}")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    process_phone_data(REALIZD_FOLDER, OUTPUT_CSV, MIN_SESSION_SECONDS)
