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

DATA_FILE = "data/processed/josaa_iit_cse_open_2021_2025.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cutoff_prediction_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI CUTOFF PREDICTION MODEL")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded.")
print(f"Rows: {len(df)}")


# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    by=["institute", "gender", "year"]
).reset_index(drop=True)


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

TARGET = "closing_rank"


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df = df[df["year"].isin([2021, 2022, 2023])].copy()
validation_df = df[df["year"] == 2024].copy()
test_df = df[df["year"] == 2025].copy()


print("\n" + "=" * 70)
print("YEAR-BASED DATA SPLIT")
print("=" * 70)

print(
    f"Training:   {len(train_df)} rows "
    f"(2021-2023)"
)

print(
    f"Validation: {len(validation_df)} rows "
    f"(2024)"
)

print(
    f"Testing:    {len(test_df)} rows "
    f"(2025)"
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# CATEGORICAL FEATURES
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
]


# ============================================================
# PREPROCESSOR
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
print("TRAINING")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(name, X, y):

    predictions = pipeline.predict(X)

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = mean_squared_error(
        y,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions
    )

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
print("VALIDATION - 2024")
print("=" * 70)

validation_predictions = evaluate_model(
    "2024 Validation",
    X_validation,
    y_validation,
)


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 70)
print("TEST - 2025")
print("=" * 70)

test_predictions = evaluate_model(
    "2025 Test",
    X_test,
    y_test,
)


# ============================================================
# SAMPLE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE TEST PREDICTIONS")
print("=" * 70)

results = test_df[
    [
        "year",
        "institute",
        "gender",
        "closing_rank",
    ]
].copy()

results["predicted_closing_rank"] = (
    test_predictions
    .round(0)
    .astype(int)
)

results["absolute_error"] = (
    results["closing_rank"]
    - results["predicted_closing_rank"]
).abs()


print(
    results
    .head(15)
    .to_string(index=False)
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_FILE
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Model file: {MODEL_FILE}"
)

print("\nCutoff prediction model completed successfully.")