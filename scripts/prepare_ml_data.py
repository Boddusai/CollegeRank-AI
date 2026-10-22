import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = Path(
    "data/processed/josaa_iit_cse_open_2021_2025.csv"
)

OUTPUT_DIR = Path("data/processed")

TRAIN_FILE = OUTPUT_DIR / "train_2021_2023.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation_2024.csv"
TEST_FILE = OUTPUT_DIR / "test_2025.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("=" * 70)
print("PREPARING ML DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nFull dataset:")
print("Rows:", len(df))


# --------------------------------------------------
# Year-based split
# --------------------------------------------------

train_df = df[df["year"].isin([2021, 2022, 2023])].copy()

validation_df = df[df["year"] == 2024].copy()

test_df = df[df["year"] == 2025].copy()


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)

validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("YEAR-BASED SPLIT")
print("=" * 70)

print("\nTraining data:")
print("Years: 2021, 2022, 2023")
print("Rows:", len(train_df))

print("\nValidation data:")
print("Year: 2024")
print("Rows:", len(validation_df))

print("\nTesting data:")
print("Year: 2025")
print("Rows:", len(test_df))


# --------------------------------------------------
# Verify years
# --------------------------------------------------

print("\n" + "=" * 70)
print("VERIFYING YEARS")
print("=" * 70)

print("\nTraining years:")
print(sorted(train_df["year"].unique()))

print("\nValidation years:")
print(sorted(validation_df["year"].unique()))

print("\nTesting years:")
print(sorted(test_df["year"].unique()))


# --------------------------------------------------
# Check for overlap
# --------------------------------------------------

train_years = set(train_df["year"])
validation_years = set(validation_df["year"])
test_years = set(test_df["year"])

print("\n" + "=" * 70)
print("CHECKING YEAR OVERLAP")
print("=" * 70)

print(
    "Train ∩ Validation:",
    train_years.intersection(validation_years)
)

print(
    "Train ∩ Test:",
    train_years.intersection(test_years)
)

print(
    "Validation ∩ Test:",
    validation_years.intersection(test_years)
)


print("\n" + "=" * 70)
print("ML DATA PREPARATION COMPLETE")
print("=" * 70)