import pandas as pd
import glob
import os

# Directory where your OHLCV CSVs are stored
data_dir = "/home/morgan/bot_project/project-bolt-sb1-7pbhnqtj/project_backups/backup_20250420_020445/data/raw"
output_csv = "/home/morgan/bot_project/project-bolt-sb1-7pbhnqtj/project_backups/backup_20250420_020445/data/merged_ohlcv.csv"

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

dfs = []
for file in csv_files:
    # Infer asset symbol from filename, e.g., "BTC_USD.csv" -> "BTC_USD"
    asset = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    df['asset'] = asset  # Add asset column
    dfs.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Optional: sort by timestamp and asset
#merged_df = merged_df.sort_values(['timestamp', 'asset'])

# Save to CSV
merged_df.to_csv(output_csv, index=False)
print(f"Merged CSV saved to {output_csv}")
