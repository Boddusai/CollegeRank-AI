import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/josaa_iit_cse_open_2021_2025.csv"

MODEL_DIR = "models"
MODEL_FILE = os.path.join(
    MODEL_DIR,
    "cutoff_trend_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI HISTORICAL TREND MODEL")
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
# FUNCTION TO PREDICT ONE COLLEGE/GENDER
# ============================================================

def predict_trend(group, target_year):

    X = group[["year"]]
    y = group["closing_rank"]

    model = LinearRegression()

    model.fit(X, y)

    prediction = model.predict(
        pd.DataFrame({"year": [target_year]})
    )[0]

    return max(1, round(prediction))


# ============================================================
# BACKTEST
# ============================================================

print("\n" + "=" * 70)
print("BACKTESTING TREND MODEL")
print("=" * 70)

actual_values = []
predicted_values = []


# ------------------------------------------------------------
# Predict 2024 using 2021-2023
# ------------------------------------------------------------

print("\n2024 Validation")
print("-" * 70)

train_2024 = df[
    df["year"].isin([2021, 2022, 2023])
].copy()

actual_2024 = df[
    df["year"] == 2024
].copy()


for _, row in actual_2024.iterrows():

    group = train_2024[
        (train_2024["institute"] == row["institute"]) &
        (train_2024["gender"] == row["gender"])
    ]

    prediction = predict_trend(
        group,
        2024
    )

    actual_values.append(
        row["closing_rank"]
    )

    predicted_values.append(
        prediction
    )


mae_2024 = mean_absolute_error(
    actual_values,
    predicted_values
)

rmse_2024 = mean_squared_error(
    actual_values,
    predicted_values
) ** 0.5

r2_2024 = r2_score(
    actual_values,
    predicted_values
)


print(f"MAE :  {mae_2024:.2f}")
print(f"RMSE:  {rmse_2024:.2f}")
print(f"R2  :  {r2_2024:.4f}")


# ============================================================
# 2025 TEST
# ============================================================

print("\n2025 Test")
print("-" * 70)

actual_values_2025 = []
predicted_values_2025 = []


train_2025 = df[
    df["year"].isin([2021, 2022, 2023, 2024])
].copy()

actual_2025 = df[
    df["year"] == 2025
].copy()


for _, row in actual_2025.iterrows():

    group = train_2025[
        (train_2025["institute"] == row["institute"]) &
        (train_2025["gender"] == row["gender"])
    ]

    prediction = predict_trend(
        group,
        2025
    )

    actual_values_2025.append(
        row["closing_rank"]
    )

    predicted_values_2025.append(
        prediction
    )


mae_2025 = mean_absolute_error(
    actual_values_2025,
    predicted_values_2025
)

rmse_2025 = mean_squared_error(
    actual_values_2025,
    predicted_values_2025
) ** 0.5

r2_2025 = r2_score(
    actual_values_2025,
    predicted_values_2025
)


print(f"MAE :  {mae_2025:.2f}")
print(f"RMSE:  {rmse_2025:.2f}")
print(f"R2  :  {r2_2025:.4f}")


# ============================================================
# TRAIN FINAL MODELS FOR 2026
# ============================================================

print("\n" + "=" * 70)
print("2026 CUTOFF PREDICTIONS")
print("=" * 70)


predictions_2026 = []


for (institute, gender), group in df.groupby(
    ["institute", "gender"]
):

    model = LinearRegression()

    X = group[["year"]]
    y = group["closing_rank"]

    model.fit(X, y)

    prediction = model.predict(
        pd.DataFrame({"year": [2026]})
    )[0]

    prediction = max(
        1,
        round(prediction)
    )

    predictions_2026.append({
        "institute": institute,
        "gender": gender,
        "predicted_2026_closing_rank": prediction
    })


predictions_df = pd.DataFrame(
    predictions_2026
)


# ============================================================
# DISPLAY 2026 PREDICTIONS
# ============================================================

print(
    predictions_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL DATA
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    predictions_df,
    MODEL_FILE
)


print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Model file: {MODEL_FILE}"
)

print("\nHistorical trend model completed successfully.")