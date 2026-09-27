import os
import pandas as pd
from config import SLEEP_META_FOLDER

# MODIFY THIS PATH
OUTPUT_FILE = "sleep_metadata_cleaned.csv"

def clean_sleep_metadata(folder, output_file):

    keep_cols = [
        "sleepId", "timeInBed", "minutesAsleep", "efficiency",
        "endTime", "startTime", "dateOfSleep", "isMainSleep"
    ]

    all_rows = []

    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue

        fp = os.path.join(folder, filename)
        df = pd.read_csv(fp)

        # extract participant ID from filename
        participant_id = os.path.splitext(filename)[0]

        # keep only columns that exist
        cols_present = [c for c in keep_cols if c in df.columns]
        sub = df[cols_present].copy()

        # filter: keep only main sleep sessions
        if "isMainSleep" in sub.columns:
            # normalize case: could be lowercase true/false or 1/0
            sub['isMainSleep'] = sub['isMainSleep'].astype(str).str.lower()
            sub = sub[sub['isMainSleep'] == 'true']
        else:
            # if no flag exists, skip this participant file
            continue

        # rename dateOfSleep -> date
        if "dateOfSleep" in sub.columns:
            sub = sub.rename(columns={"dateOfSleep": "date"})

        sub["participant_id"] = participant_id

        all_rows.append(sub)

    if not all_rows:
        print("No valid main sleep rows found.")
        return

    final_df = pd.concat(all_rows, ignore_index=True)

    # reorder columns for clean final CSV
    final_df = final_df[[
        "participant_id","sleepId","date","startTime","endTime",
        "timeInBed","minutesAsleep","efficiency"
    ]]

    final_df.to_csv(output_file, index=False)
    print("✅ clean sleep metadata saved:", output_file)
    print("participants:", final_df["participant_id"].nunique(), "  rows:", len(final_df))

if __name__ == "__main__":
    clean_sleep_metadata(SLEEP_META_FOLDER, OUTPUT_FILE)
