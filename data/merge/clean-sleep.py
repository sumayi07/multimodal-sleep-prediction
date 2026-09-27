import pandas as pd

# ==== EDIT THESE IF NEEDED ====
sleep_metadata_file = "sleep_metadata_cleaned.csv"
sleep_stage_file    = "sleepdata_cleaned.csv"
output_file         = "sleep_master.csv"  
# ===============================

# load both
meta = pd.read_csv(sleep_metadata_file)
stages = pd.read_csv(sleep_stage_file)

# rename columns inside stages file for final merge naming consistency
stages = stages.rename(columns={
    "remDurationMinutes": "remDuration",
    "deepDurationMinutes": "deepDuration"
})

# merge using BOTH participant_id and sleep_id
merged = pd.merge(
    meta,
    stages[["participant_id", "sleepId", "remDuration", "deepDuration"]],
    on=["participant_id", "sleepId"],
    how="left"
)

# save
merged.to_csv(output_file, index=False)
print("✅ merged sleep metadata + stage durations")
print("saved to:", output_file)
print("Rows:", len(merged))
print("Columns:", list(merged.columns))
