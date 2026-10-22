import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_FILE = Path("models/final_xgboost_model.pkl")
TEST_FILE = Path("data/processed/test_2025.csv")


print("=" * 70)
print("COLLEGERANK-AI - TEST FINAL XGBOOST MODEL")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading saved model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_df = pd.read_csv(TEST_FILE)

print("\nTest dataset loaded.")
print("Rows:", len(test_df))
print("Year:", sorted(test_df["year"].unique()))


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "year",
    "round",
    "institute_type",
    "institute",
    "academic_program",
    "quota",
    "seat_type",
    "gender",
]


X_test = test_df[FEATURES]


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)


# ============================================================
# DISPLAY SAMPLE PREDICTIONS
# ============================================================

result_df = test_df[
    [
        "year",
        "institute",
        "gender",
        "opening_rank",
        "closing_rank",
    ]
].copy()

result_df["predicted_closing_rank"] = predictions.round(2)

result_df["absolute_error"] = (
    result_df["closing_rank"]
    - result_df["predicted_closing_rank"]
).abs().round(2)


print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

print(
    result_df.head(10).to_string(index=False)
)


# ============================================================
# BASIC ERROR CHECK
# ============================================================

mae = result_df["absolute_error"].mean()

print("\n" + "=" * 70)
print("PREDICTION CHECK")
print("=" * 70)

print(f"\nMean Absolute Error on 2025 test data: {mae:.2f}")

print("\nPrediction range:")
print(f"Minimum predicted closing rank: {predictions.min():.2f}")
print(f"Maximum predicted closing rank: {predictions.max():.2f}")


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("FINAL XGBOOST MODEL TEST COMPLETED")
print("=" * 70)