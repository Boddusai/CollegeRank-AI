import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/final_cutoff_predictions_2026.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("COLLEGERANK-AI — JEE COLLEGE RECOMMENDATION SYSTEM")
print("=" * 70)


# ============================================================
# GET STUDENT RANK
# ============================================================

while True:

    try:

        student_rank = int(
            input("\nEnter your JEE Advanced rank: ")
        )

        if student_rank <= 0:

            print("Rank must be greater than 0.")
            continue

        break

    except ValueError:

        print("Please enter a valid integer rank.")


# ============================================================
# GET GENDER
# ============================================================

print("\nSelect gender category:")

print("1. Gender-Neutral")
print("2. Female-only (including Supernumerary)")

while True:

    gender_choice = input(
        "\nEnter choice (1 or 2): "
    ).strip()

    if gender_choice == "1":

        gender = "Gender-Neutral"
        break

    elif gender_choice == "2":

        gender = "Female-only (including Supernumerary)"
        break

    else:

        print("Please enter 1 or 2.")


# ============================================================
# FILTER GENDER
# ============================================================

gender_df = df[
    df["gender"] == gender
].copy()


# ============================================================
# COMPARE RANK WITH PREDICTED CUTOFF
# ============================================================

gender_df["rank_difference"] = (
    gender_df["final_predicted_2026_cutoff"]
    - student_rank
)


# ============================================================
# CLASSIFY COLLEGES
# ============================================================

def classify_college(row):

    cutoff = row["final_predicted_2026_cutoff"]

    difference = cutoff - student_rank

    # Student rank is comfortably better than
    # predicted closing rank.
    if difference >= 500:

        return "Comfortable Range"

    # Student rank is better than or equal to
    # predicted closing rank.
    elif difference >= 0:

        return "Within Predicted Cutoff"

    # Student rank is slightly worse than
    # predicted cutoff.
    elif difference >= -500:

        return "Borderline"

    # Student rank is substantially worse.
    else:

        return "Outside Predicted Cutoff"


gender_df["category"] = (
    gender_df.apply(
        classify_college,
        axis=1
    )
)


# ============================================================
# SORT BY PREDICTED CUTOFF
# ============================================================

gender_df = gender_df.sort_values(
    "final_predicted_2026_cutoff",
    ascending=True
)


# ============================================================
# DISPLAY STUDENT DETAILS
# ============================================================

print("\n" + "=" * 70)
print("STUDENT DETAILS")
print("=" * 70)

print(f"JEE Advanced Rank : {student_rank}")
print(f"Gender Category   : {gender}")


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 70)
print("COLLEGE RECOMMENDATIONS")
print("=" * 70)


display_columns = [
    "institute",
    "final_predicted_2026_cutoff",
    "rank_difference",
    "category"
]


print(
    gender_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

comfortable = len(
    gender_df[
        gender_df["category"]
        == "Comfortable Range"
    ]
)

within_cutoff = len(
    gender_df[
        gender_df["category"]
        == "Within Predicted Cutoff"
    ]
)

borderline = len(
    gender_df[
        gender_df["category"]
        == "Borderline"
    ]
)

outside = len(
    gender_df[
        gender_df["category"]
        == "Outside Predicted Cutoff"
    ]
)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Comfortable Range       : {comfortable}")
print(f"Within Predicted Cutoff: {within_cutoff}")
print(f"Borderline              : {borderline}")
print(f"Outside Predicted Cutoff: {outside}")


# ============================================================
# IMPORTANT NOTICE
# ============================================================

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "These are model-based 2026 cutoff estimates, "
    "not guaranteed admission cutoffs."
)

print(
    "Actual JoSAA 2026 cutoffs may differ because of "
    "seat availability, preferences, competition, "
    "and other admission factors."
)

print("\nRecommendation process completed.")