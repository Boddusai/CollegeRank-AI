import pandas as pd
from pathlib import Path


# Project paths
PROCESSED_DIR = Path("data/processed")

# Historical JoSAA files to combine
files = [
    PROCESSED_DIR / "josaa_2021_round2_iit_cse_open.csv",
    PROCESSED_DIR / "josaa_2022_round2_iit_cse_open.csv",
    PROCESSED_DIR / "josaa_2023_round2_iit_cse_open.csv",
    PROCESSED_DIR / "josaa_2024_round2_iit_cse_open.csv",
    PROCESSED_DIR / "josaa_2025_round2_iit_cse_open.csv",
]

all_data = []

print("=" * 70)
print("COMBINING JOSAA HISTORICAL DATA")
print("=" * 70)

for file in files:
    print(f"\nReading: {file}")

    if not file.exists():
        print("ERROR: File not found!")
        continue

    df = pd.read_csv(file)

    print(f"Rows found: {len(df)}")

    all_data.append(df)


# Combine all yearly datasets
combined_df = pd.concat(all_data, ignore_index=True)


# Make sure numeric columns are numeric
combined_df["opening_rank"] = pd.to_numeric(
    combined_df["opening_rank"],
    errors="coerce"
)

combined_df["closing_rank"] = pd.to_numeric(
    combined_df["closing_rank"],
    errors="coerce"
)


# Add a useful cutoff range feature
combined_df["cutoff_range"] = (
    combined_df["closing_rank"] - combined_df["opening_rank"]
)


# Save master dataset
output_file = (
    PROCESSED_DIR / "josaa_iit_cse_open_2021_2025.csv"
)

combined_df.to_csv(output_file, index=False)


print("\n" + "=" * 70)
print("COMBINATION COMPLETE")
print("=" * 70)

print(f"Total rows: {len(combined_df)}")
print(f"Total columns: {len(combined_df.columns)}")

print("\nColumns:")
print(list(combined_df.columns))

print("\nRows by year:")
print(combined_df["year"].value_counts().sort_index())

print("\nMissing values:")
print(combined_df.isnull().sum())

print(f"\nMaster dataset saved:")
print(output_file)