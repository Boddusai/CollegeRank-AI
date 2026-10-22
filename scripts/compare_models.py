import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/josaa_iit_cse_open_2021_2025.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI MODEL COMPARISON")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded.")
print(f"Rows: {len(df)}")


# ============================================================
# TREND PREDICTION FUNCTION
# ============================================================

def predict_trend(group, target_year):

    X = group[["year"]]
    y = group["closing_rank"]

    model = LinearRegression()

    model.fit(X, y)

    future_year = pd.DataFrame(
        {"year": [target_year]}
    )

    prediction = model.predict(
        future_year
    )[0]

    return max(1, round(prediction))


# ============================================================
# EVALUATE TREND MODEL
# ============================================================

def evaluate_trend(train_years, test_year):

    train_df = df[
        df["year"].isin(train_years)
    ].copy()

    test_df = df[
        df["year"] == test_year
    ].copy()

    actual = []
    predicted = []

    for _, row in test_df.iterrows():

        group = train_df[
            (train_df["institute"] == row["institute"]) &
            (train_df["gender"] == row["gender"])
        ]

        prediction = predict_trend(
            group,
            test_year
        )

        actual.append(
            row["closing_rank"]
        )

        predicted.append(
            prediction
        )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = mean_squared_error(
        actual,
        predicted
    ) ** 0.5

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# 2024 VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("2024 VALIDATION")
print("=" * 70)

mae_2024, rmse_2024, r2_2024 = evaluate_trend(
    [2021, 2022, 2023],
    2024
)

print(f"MAE :  {mae_2024:.2f}")
print(f"RMSE:  {rmse_2024:.2f}")
print(f"R2  :  {r2_2024:.4f}")


# ============================================================
# 2025 TEST
# ============================================================

print("\n" + "=" * 70)
print("2025 TEST")
print("=" * 70)

mae_2025, rmse_2025, r2_2025 = evaluate_trend(
    [2021, 2022, 2023, 2024],
    2025
)

print(f"MAE :  {mae_2025:.2f}")
print(f"RMSE:  {rmse_2025:.2f}")
print(f"R2  :  {r2_2025:.4f}")


# ============================================================
# SIMPLE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("SIMPLE BASELINE")
print("=" * 70)

train_df = df[
    df["year"].isin([2021, 2022, 2023, 2024])
].copy()

test_df = df[
    df["year"] == 2025
].copy()

baseline_actual = []
baseline_predicted = []


for _, row in test_df.iterrows():

    group = train_df[
        (train_df["institute"] == row["institute"]) &
        (train_df["gender"] == row["gender"])
    ]

    # Use the latest available cutoff as prediction
    latest_cutoff = (
        group
        .sort_values("year")
        .iloc[-1]["closing_rank"]
    )

    baseline_actual.append(
        row["closing_rank"]
    )

    baseline_predicted.append(
        latest_cutoff
    )


baseline_mae = mean_absolute_error(
    baseline_actual,
    baseline_predicted
)

baseline_rmse = mean_squared_error(
    baseline_actual,
    baseline_predicted
) ** 0.5

baseline_r2 = r2_score(
    baseline_actual,
    baseline_predicted
)


print(f"MAE :  {baseline_mae:.2f}")
print(f"RMSE:  {baseline_rmse:.2f}")
print(f"R2  :  {baseline_r2:.4f}")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Historical Trend",
        "Latest Cutoff Baseline"
    ],
    "MAE": [
        mae_2025,
        baseline_mae
    ],
    "RMSE": [
        rmse_2025,
        baseline_rmse
    ],
    "R2": [
        r2_2025,
        baseline_r2
    ]
})

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 2026 PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("2026 TREND PREDICTIONS")
print("=" * 70)

predictions = []

for (institute, gender), group in df.groupby(
    ["institute", "gender"]
):

    prediction = predict_trend(
        group,
        2026
    )

    latest_cutoff = (
        group
        .sort_values("year")
        .iloc[-1]["closing_rank"]
    )

    predictions.append({
        "institute": institute,
        "gender": gender,
        "latest_2025_cutoff": int(latest_cutoff),
        "predicted_2026_cutoff": int(prediction),
        "change_from_2025": int(
            prediction - latest_cutoff
        )
    })


predictions_df = pd.DataFrame(
    predictions
)

print(
    predictions_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "data/processed/"
    "cutoff_predictions_2026.csv"
)

predictions_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)

print(
    f"File: {output_file}"
)

print("\nModel comparison completed successfully.")