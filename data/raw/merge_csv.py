import pandas as pd
import glob
import os

def merge_ohlcv_files(directory="."):
    """
    Merges CSV files with 'date', 'open', 'high', 'low', and 'close' headers
    in a given directory. Adds an 'asset' column derived from the filename.

    Args:
        directory (str, optional): The directory containing the CSV files.
                                     Defaults to the current directory.

    Returns:
        pandas.DataFrame: A DataFrame containing the merged data with the
                          added 'asset' column. Returns an empty DataFrame
                          if no matching files are found.
    """
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    all_df = []

    for f in all_files:
        try:
            df = pd.read_csv(f)
            if all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close']):
                filename = os.path.basename(f)
                asset_name = os.path.splitext(filename)[0].replace("_ohlvc", "").replace("_", "/")
                df['asset'] = asset_name
                all_df.append(df)
            else:
                print(f"Skipping '{filename}': Missing required columns ('date', 'open', 'high', 'low', 'close').")
        except Exception as e:
            print(f"Error reading file '{os.path.basename(f)}': {e}")

    if all_df:
        merged_df = pd.concat(all_df, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    merged_data = merge_ohlcv_files()

    if not merged_data.empty:
        print("Successfully merged CSV files.")
        print(merged_data.head())  # Display the first few rows of the merged data
        # You can save the merged data to a new CSV file if needed:
        merged_data.to_csv("merged_data.csv", index=False)
    else:
        print("No valid CSV files found to merge.")
