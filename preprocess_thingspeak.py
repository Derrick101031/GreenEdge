# preprocess_thingspeak.py (v1.1 - More Robust)
# This script reads the raw CSV from ThingSpeak, validates the columns,
# and prepares it for the model notebook.

import pandas as pd
import os

# --- Configuration ---
COLUMN_MAPPING = {
    'created_at': 'timestamp',
    'field1': 'soil_moisture',
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
        print(f"Error: Input file '{INPUT_FILENAME}' not found. Did the download fail?")
        exit(1)

    print(f"Reading raw data from {INPUT_FILENAME}...")
    try:
        df = pd.read_csv(INPUT_FILENAME)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("The file might be empty or not in CSV format. Check ThingSpeak API key and channel ID.")
        exit(1)

    # --- NEW: Robust column handling ---
    # Find which of the required columns actually exist in the downloaded file
    available_columns = [col for col in COLUMN_MAPPING.keys() if col in df.columns]
    
    if not available_columns:
        print("Error: None of the expected columns (created_at, field1, etc.) were found in the downloaded data.")
        print("Please verify your THINGSPEAK_CHANNEL_ID and THINGSPEAK_READ_API_KEY secrets.")
        exit(1)

    print(f"Found available columns: {available_columns}")

    # Keep only the columns that are available
    df = df[available_columns]

    # Rename the available columns to match the notebook's expectations
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    
    # The notebook expects a specific date format, so let's set the index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    # Convert all data columns to numeric, coercing errors
    for col in df.columns:
        if col != 'timestamp': # Don't try to convert the index
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with any missing data after conversion
    df.dropna(inplace=True)

    if df.empty:
        print("Warning: The final DataFrame is empty after cleaning. Not enough valid data was fetched.")
        # We can either exit or create an empty dataset.csv
        # For this use case, we'll exit to make the issue clear in the logs.
        exit(1)

    print(f"Saving pre-processed data to {OUTPUT_FILENAME}...")
    df.to_csv(OUTPUT_FILENAME)

    print("Pre-processing complete.")
