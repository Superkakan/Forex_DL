import os
import pandas as pd

# Set up paths
current_path = os.getcwd()
data_path = os.path.join(current_path, "data", "eurusd")

print("\nData path:", data_path, "\n")

# List only CSV files
files = [file for file in os.listdir(data_path) if file.endswith(".csv")]

# Column names based on data description
col_names = ["date", "time", "open", "high", "low", "close", "volume"]

df_list = []

# Read each file and assign column names
for file in files:
    file_path = os.path.join(data_path, file)
    
    try:
        df = pd.read_csv(file_path, header=None, sep=',', names=col_names)
    except pd.errors.ParserError:
        df = pd.read_csv(file_path, header=None, sep=';', names=col_names)
        # Some files might use commas
    
    df_list.append(df)

# Concatenate all dataframes
merged_df = pd.concat(df_list, ignore_index=True)

# Create datetime column (optional but recommended)
merged_df["datetime"] = pd.to_datetime(merged_df["date"] + " " + merged_df["time"], format="%Y.%m.%d %H:%M", errors="coerce")

# Drop rows where datetime conversion failed (optional)
merged_df.dropna(subset=["datetime"], inplace=True)

# Reset index after dropping rows
merged_df.reset_index(drop=True, inplace=True)

# Print result
print(merged_df.head())
print(f"\nTotal rows merged: {len(merged_df)}")


merged_df.to_csv("merged_eurusd.csv")

