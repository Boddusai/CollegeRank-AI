import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/processed/train_2021_2023.csv"
VALIDATION_FILE = "data/processed/validation_2024.csv"
TEST_FILE = "data/processed/test_2025.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "college_rank_model.pkl")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI MODEL TRAINING")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nDatasets loaded successfully.")

print(f"Training rows:    {len(train_df)}")
print(f"Validation rows:  {len(validation_df)}")
print(f"Testing rows:     {len(test_df)}")


# ============================================================
# TARGET
# ============================================================

TARGET = "closing_rank"


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
    "opening_rank",
]


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_features = [
    "institute_type",
    "institute",
    "academic_program",
    "quota",
    "seat_type",
    "gender",
]

numeric_features = [
    "year",
    "round",
    "opening_rank",
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

pipeline.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, X, y):
    predictions = pipeline.predict(X)

    mae = mean_absolute_error(y, predictions)
    rmse = mean_squared_error(y, predictions) ** 0.5
    r2 = r2_score(y, predictions)

    print(f"\n{name}")
    print("-" * 70)
    print(f"MAE :  {mae:.2f}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"R2  :  {r2:.4f}")

    return predictions


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULTS - 2024")
print("=" * 70)

validation_predictions = evaluate_model(
    "Validation Dataset",
    X_validation,
    y_validation,
)


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS - 2025")
print("=" * 70)

test_predictions = evaluate_model(
    "Test Dataset",
    X_test,
    y_test,
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TEST PREDICTIONS")
print("=" * 70)

results = test_df[
    [
        "year",
        "institute",
        "gender",
        "opening_rank",
        "closing_rank",
    ]
].copy()

results["predicted_closing_rank"] = test_predictions.round(0).astype(int)

print(results.head(10).to_string(index=False))


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, MODEL_FILE)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"Model file: {MODEL_FILE}")
print("\nTraining pipeline completed successfully.")