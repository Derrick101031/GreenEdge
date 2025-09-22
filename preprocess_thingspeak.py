# preprocess_thingspeak.py
# This script reads the raw CSV from ThingSpeak and prepares it for the model notebook.

import pandas as pd
import os

# --- Configuration ---
# This mapping is based on the fields sent in dt.ino [cite: 96]
# and the column names expected in model.ipynb
COLUMN_MAPPING = {
    'created_at': 'timestamp',
    'field1': 'soil_moisture', # Sent as vwc in the .ino file
    'field2': 'soil_temp',
    'field3': 'ec',
    'field4': 'ph',
    'field5': 'ambient_temp',
    'field6': 'ambient_hum'
}
INPUT_FILENAME = "thingspeak_feed.csv"
OUTPUT_FILENAME = "dataset.csv"

# --- Main Script ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: Input file '{INPUT_FILENAME}' not found. Skipping pre-processing.")
        exit()

    print(f"Reading raw data from {INPUT_FILENAME}...")
    df = pd.read_csv(INPUT_FILENAME)

    # Keep only the columns we need
    df = df[list(COLUMN_MAPPING.keys())]

    # Rename columns to match the notebook's expectations
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # The notebook expects a specific date format, so let's set the index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Convert all data columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any non-numeric data
    df.dropna(inplace=True)

    print(f"Saving pre-processed data to {OUTPUT_FILENAME}...")
    df.to_csv(OUTPUT_FILENAME)

    print("Pre-processing complete.")
