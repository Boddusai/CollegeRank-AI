import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/processed/josaa_iit_cse_open_2021_2025.csv")

print("=" * 70)
print("JOSAA DATASET EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")

# --------------------------------------------------
# Basic information
# --------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# --------------------------------------------------
# Columns
# --------------------------------------------------

print("\n" + "=" * 70)
print("2. COLUMNS")
print("=" * 70)

for column in df.columns:
    print("-", column)

# --------------------------------------------------
# Data types
# --------------------------------------------------

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(df.dtypes)

# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\n" + "=" * 70)
print("4. MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())

# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------

print("\n" + "=" * 70)
print("5. DUPLICATE ROWS")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())

# --------------------------------------------------
# Years
# --------------------------------------------------

print("\n" + "=" * 70)
print("6. ROWS BY YEAR")
print("=" * 70)

print(df["year"].value_counts().sort_index())

# --------------------------------------------------
# Institutes
# --------------------------------------------------

print("\n" + "=" * 70)
print("7. NUMBER OF IITs")
print("=" * 70)

print("Unique IITs:", df["institute"].nunique())

# --------------------------------------------------
# Gender
# --------------------------------------------------

print("\n" + "=" * 70)
print("8. GENDER CATEGORIES")
print("=" * 70)

print(df["gender"].value_counts())

# --------------------------------------------------
# Seat type
# --------------------------------------------------

print("\n" + "=" * 70)
print("9. SEAT TYPES")
print("=" * 70)

print(df["seat_type"].value_counts())

# --------------------------------------------------
# Quota
# --------------------------------------------------

print("\n" + "=" * 70)
print("10. QUOTA")
print("=" * 70)

print(df["quota"].value_counts())

# --------------------------------------------------
# Rank statistics
# --------------------------------------------------

print("\n" + "=" * 70)
print("11. RANK STATISTICS")
print("=" * 70)

print(
    df[
        [
            "opening_rank",
            "closing_rank",
            "cutoff_range"
        ]
    ].describe()
)

# --------------------------------------------------
# Cutoff by year
# --------------------------------------------------

print("\n" + "=" * 70)
print("12. AVERAGE CLOSING RANK BY YEAR")
print("=" * 70)

print(
    df.groupby("year")["closing_rank"]
    .mean()
    .round(2)
)

# --------------------------------------------------
# Cutoff by IIT
# --------------------------------------------------

print("\n" + "=" * 70)
print("13. AVERAGE CLOSING RANK BY IIT")
print("=" * 70)

print(
    df.groupby("institute")["closing_rank"]
    .mean()
    .sort_values()
    .round(2)
)

# --------------------------------------------------
# Dataset preview
# --------------------------------------------------

print("\n" + "=" * 70)
print("14. FIRST 10 ROWS")
print("=" * 70)

print(df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)