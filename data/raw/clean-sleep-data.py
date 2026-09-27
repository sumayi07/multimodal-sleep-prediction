import os
import pandas as pd
from config import SLEEP_DATA_FOLDER

# ==== EDIT THESE PATHS ====
OUTPUT_CSV = "sleepdata_cleaned.csv"
# ==========================

def normalize_columns(df):
    """Make column names case-insensitive and map common variants."""
    lower = {c.lower(): c for c in df.columns}

    def pick(*cands, required=False):
        for c in cands:
            if c in lower:
                return lower[c]
        if required:
            raise KeyError(f"Missing required column. Tried any of: {cands}")
        return None

    cols = {
        "sleepId": pick("sleepid", required=True),
        "dateTime": pick("datetime", "dateTime", "date_time", required=True),
        "level": pick("level", "stage", required=True),
        "seconds": pick("seconds", "duration", "secs", required=True),
    }
    return cols

def clean_sleepdata(folder, output_csv):
    all_parts = []

    for filename in os.listdir(folder):
        if not filename.endswith(".csv"):
            continue
        fp = os.path.join(folder, filename)

        try:
            df = pd.read_csv(fp)
            cols = normalize_columns(df)

            # Extract participant_id from filename (without extension)
            participant_id = os.path.splitext(filename)[0]

            # Work on a normalized view
            sub = df[[cols["sleepId"], cols["dateTime"], cols["level"], cols["seconds"]]].copy()
            sub.rename(
                columns={
                    cols["sleepId"]: "sleepId",
                    cols["dateTime"]: "dateTime",
                    cols["level"]: "level",
                    cols["seconds"]: "seconds",
                },
                inplace=True,
            )

            # Parse datetime & normalize
            sub["dateTime"] = pd.to_datetime(sub["dateTime"], errors="coerce")
            # Lowercase level for matching
            sub["level"] = sub["level"].astype(str).str.lower()

            # Keep only rows with valid time & seconds
            sub = sub.dropna(subset=["dateTime"])
            sub = sub[pd.to_numeric(sub["seconds"], errors="coerce").notna()]
            sub["seconds"] = sub["seconds"].astype(float)

            if sub.empty:
                continue

            # For each sleepId, we want:
            # - date = date of the first timestamp (min dateTime) -> date only (YYYY-MM-DD)
            # - total REM seconds (sum where level == 'rem')
            # - total Deep seconds (sum where level == 'deep')
            first_dt = sub.groupby("sleepId")["dateTime"].min().rename("first_dt")
            sums = sub.pivot_table(
                index="sleepId",
                columns="level",
                values="seconds",
                aggfunc="sum",
                fill_value=0.0,
            )

            # Make sure missing columns appear
            for needed in ["rem", "deep"]:
                if needed not in sums.columns:
                    sums[needed] = 0.0

            out = sums[["rem", "deep"]].reset_index()
            out = out.merge(first_dt.reset_index(), on="sleepId", how="left")
            out["date"] = out["first_dt"].dt.date
            out.drop(columns=["first_dt"], inplace=True)

            # Convert seconds -> minutes
            out["remDurationMinutes"] = (out["rem"] / 60.0)
            out["deepDurationMinutes"] = (out["deep"] / 60.0)
            out.drop(columns=["rem", "deep"], inplace=True)

            # Add participant_id
            out["participant_id"] = participant_id

            # Reorder columns
            out = out[["participant_id", "sleepId", "date", "remDurationMinutes", "deepDurationMinutes"]]

            all_parts.append(out)

        except Exception as e:
            print(f"⚠️ Skipping {filename}: {e}")

    if not all_parts:
        print("No valid sleep-data found. Check folder and columns.")
        return

    final_df = pd.concat(all_parts, ignore_index=True).drop_duplicates()

    # Optional: round to, e.g., 2 decimal places (remove if you prefer raw float)
    final_df["remDurationMinutes"] = final_df["remDurationMinutes"].round(2)
    final_df["deepDurationMinutes"] = final_df["deepDurationMinutes"].round(2)

    final_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")
    print(f"   Participants: {final_df['participant_id'].nunique()}  Rows (sleep sessions): {len(final_df)}")

if __name__ == "__main__":
    clean_sleepdata(SLEEP_DATA_FOLDER, OUTPUT_CSV)
