
import numpy as np
import pandas as pd


def calculate_hybrid_cutoffs(
    prediction_rows: pd.DataFrame,
    historical_data: pd.DataFrame,
    rf_predictions,
    rf_weight: float = 0.75,
) -> np.ndarray:
    """
    Combine Random Forest predictions with weighted historical cutoffs.

    Only historical records from years before the prediction year are used.
    Rows without matching historical records retain their RF prediction.

    This helper does not train, save, or modify any model.
    """
    if not 0.0 <= rf_weight <= 1.0:
        raise ValueError("rf_weight must be between 0 and 1.")

    rf = np.asarray(rf_predictions, dtype=float)

    if len(prediction_rows) != len(rf):
        raise ValueError(
            "prediction_rows and rf_predictions must have equal lengths."
        )

    required = [
        "exam",
        "year",
        "round",
        "institute_type",
        "institute",
        "academic_program",
        "quota",
        "seat_type",
        "gender",
        "closing_rank",
    ]

    missing = [
        column for column in required
        if column not in historical_data.columns
    ]

    if missing:
        raise ValueError(
            f"Historical data is missing required columns: {missing}"
        )

    prediction_required = [
        "exam",
        "year",
        "round",
        "institute_type",
        "institute",
        "academic_program",
        "quota",
        "seat_type",
        "gender",
    ]

    missing_predictions = [
        column for column in prediction_required
        if column not in prediction_rows.columns
    ]

    if missing_predictions:
        raise ValueError(
            "Prediction rows are missing required columns: "
            f"{missing_predictions}"
        )

    keys = [
        "exam",
        "round",
        "institute_type",
        "institute",
        "academic_program",
        "quota",
        "seat_type",
        "gender",
    ]

    history = historical_data[required].copy()
    history["year"] = pd.to_numeric(history["year"], errors="coerce")
    history["closing_rank"] = pd.to_numeric(
        history["closing_rank"], errors="coerce"
    )

    history = history.dropna(
        subset=keys + ["year", "closing_rank"]
    )

    result = np.maximum(1, np.rint(rf)).astype(int)

    for index, row in prediction_rows.reset_index(drop=True).iterrows():
        prediction_year = pd.to_numeric(
            row["year"], errors="coerce"
        )

        if pd.isna(prediction_year):
            continue

        # Never use a cutoff from the prediction year or a future year.
        eligible = history[
            history["year"] < prediction_year
        ].copy()

        for key in keys:
            eligible = eligible[
                eligible[key].astype(str) == str(row[key])
            ]

        if eligible.empty:
            continue

        # Use up to the three most recent eligible historical years.
        years = sorted(eligible["year"].unique())[-3:]
        eligible = eligible[eligible["year"].isin(years)].copy()

        weights = {
            year: position
            for position, year in enumerate(years, start=1)
        }

        eligible["weight"] = eligible["year"].map(weights)

        weighted_cutoff = (
            eligible["closing_rank"] * eligible["weight"]
        ).sum() / eligible["weight"].sum()

        hybrid = (
            rf_weight * rf[index]
            + (1.0 - rf_weight) * weighted_cutoff
        )

        result[index] = max(1, int(np.rint(hybrid)))

    return result
