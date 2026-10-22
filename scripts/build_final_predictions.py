import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/cutoff_predictions_2026.csv"

OUTPUT_FILE = "data/processed/final_cutoff_predictions_2026.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI FINAL 2026 CUTOFF PREDICTIONS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nPrediction data loaded.")
print(f"Rows: {len(df)}")


# ============================================================
# HYBRID PREDICTION
# ============================================================

# The latest 2025 cutoff is given more importance because
# our 2025 holdout test showed that this baseline performed
# better than the historical trend model.
#
# 70% latest cutoff
# 30% historical trend prediction

df["final_predicted_2026_cutoff"] = (
    0.70 * df["latest_2025_cutoff"]
    + 0.30 * df["predicted_2026_cutoff"]
)

df["final_predicted_2026_cutoff"] = (
    df["final_predicted_2026_cutoff"]
    .round()
    .astype(int)
)


# ============================================================
# EXPECTED CHANGE
# ============================================================

df["expected_change_from_2025"] = (
    df["final_predicted_2026_cutoff"]
    - df["latest_2025_cutoff"]
)


# ============================================================
# CHANGE PERCENTAGE
# ============================================================

df["expected_change_percent"] = (
    (
        df["final_predicted_2026_cutoff"]
        - df["latest_2025_cutoff"]
    )
    / df["latest_2025_cutoff"]
    * 100
)

df["expected_change_percent"] = (
    df["expected_change_percent"]
    .round(2)
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("FINAL 2026 PREDICTIONS")
print("=" * 70)

display_columns = [
    "institute",
    "gender",
    "latest_2025_cutoff",
    "predicted_2026_cutoff",
    "final_predicted_2026_cutoff",
    "expected_change_from_2025",
    "expected_change_percent"
]

print(
    df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("FINAL PREDICTIONS SAVED")
print("=" * 70)

print(f"File: {OUTPUT_FILE}")

print("\nCollegeRank-AI prediction generation completed successfully.")