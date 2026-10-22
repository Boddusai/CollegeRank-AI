import pandas as pd
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path("data/processed/train_2021_2023.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "final_xgboost_model.pkl"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI - SAVE FINAL XGBOOST MODEL")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)

print("\nTraining dataset loaded.")
print("Rows:", len(train_df))
print("Years:", sorted(train_df["year"].unique()))


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

CATEGORICAL_FEATURES = [
    "institute_type",
    "institute",
    "academic_program",
    "quota",
    "seat_type",
    "gender",
]

NUMERIC_FEATURES = [
    "year",
    "round",
]


X_train = train_df[FEATURES]
y_train = train_df[TARGET]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
    ]
)


# ============================================================
# XGBOOST MODEL
# ============================================================

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")

model_pipeline.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model_pipeline, MODEL_FILE)


# ============================================================
# VERIFY FILE
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("\nModel file:")
print(MODEL_FILE)

print("\nModel type:")
print("XGBoost Regression Pipeline")

print("\nFeatures used:")
for feature in FEATURES:
    print(" -", feature)

print("\nTraining years:")
print(sorted(train_df["year"].unique()))

print("\nTarget:")
print(TARGET)

print("\n" + "=" * 70)
print("FINAL XGBOOST MODEL READY")
print("=" * 70)