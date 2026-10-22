import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "data/processed/train_2021_2023.csv"
VALIDATION_FILE = "data/processed/validation_2024.csv"
TEST_FILE = "data/processed/test_2025.csv"

RESULT_FILE = "data/processed/ml_model_comparison.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI - THREE MODEL ML COMPARISON")
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
]


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_validation = validation_df[FEATURES]
y_validation = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# FEATURE TYPES
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

def create_preprocessor():

    return ColumnTransformer(
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
# MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
    ),
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model):

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_predictions = pipeline.predict(
        X_validation
    )

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions,
    )

    validation_rmse = mean_squared_error(
        y_validation,
        validation_predictions,
    ) ** 0.5

    validation_r2 = r2_score(
        y_validation,
        validation_predictions,
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = pipeline.predict(
        X_test
    )

    test_mae = mean_absolute_error(
        y_test,
        test_predictions,
    )

    test_rmse = mean_squared_error(
        y_test,
        test_predictions,
    ) ** 0.5

    test_r2 = r2_score(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n2024 VALIDATION")
    print("-" * 70)
    print(f"MAE :  {validation_mae:.2f}")
    print(f"RMSE:  {validation_rmse:.2f}")
    print(f"R2  :  {validation_r2:.4f}")

    print("\n2025 TEST")
    print("-" * 70)
    print(f"MAE :  {test_mae:.2f}")
    print(f"RMSE:  {test_rmse:.2f}")
    print(f"R2  :  {test_r2:.4f}")

    return {
        "Model": model_name,

        "Validation_MAE": validation_mae,
        "Validation_RMSE": validation_rmse,
        "Validation_R2": validation_r2,

        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
    }


# ============================================================
# TRAIN ALL MODELS
# ============================================================

results = []

for model_name, model in models.items():

    result = evaluate_model(
        model_name,
        model,
    )

    results.append(result)


# ============================================================
# COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY VALIDATION MAE
# ============================================================

comparison_df = comparison_df.sort_values(
    by="Validation_MAE"
).reset_index(drop=True)


# ============================================================
# DISPLAY FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL THREE-MODEL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SELECT MODEL
# ============================================================

selected_model = comparison_df.iloc[0]["Model"]

print("\n" + "=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print(
    "Selection metric: Validation MAE"
)

print(
    f"Selected model: {selected_model}"
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

os.makedirs(
    "data/processed",
    exist_ok=True,
)

comparison_df.to_csv(
    RESULT_FILE,
    index=False,
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON RESULTS SAVED")
print("=" * 70)

print(
    f"File: {RESULT_FILE}"
)

print("\nThree-model ML comparison completed successfully.")