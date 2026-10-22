import streamlit as st
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed/final_cutoff_predictions_2026.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CollegeRank-AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_FILE)


df = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎓 CollegeRank-AI")

st.sidebar.caption(
    "JEE Advanced IIT CSE Recommendation System"
)

st.sidebar.divider()

st.sidebar.header("Student Details")


student_rank = st.sidebar.number_input(
    "JEE Advanced Rank",
    min_value=1,
    max_value=100000,
    value=1500,
    step=1
)


gender = st.sidebar.selectbox(
    "Gender Category",
    [
        "Gender-Neutral",
        "Female-only (including Supernumerary)"
    ]
)


st.sidebar.divider()

st.sidebar.info(
    "Enter your JEE Advanced rank and gender "
    "to explore estimated IIT CSE cutoff ranges."
)


# ============================================================
# FILTER DATA
# ============================================================

gender_df = df[
    df["gender"] == gender
].copy()


# ============================================================
# CALCULATE DIFFERENCE
# ============================================================

gender_df["rank_difference"] = (
    gender_df["final_predicted_2026_cutoff"]
    - student_rank
)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_college(row):

    difference = row["rank_difference"]

    if difference >= 500:
        return "Strong Match"

    elif difference >= 0:
        return "Within Predicted Cutoff"

    elif difference >= -500:
        return "Borderline"

    else:
        return "Outside Predicted Cutoff"


gender_df["category"] = gender_df.apply(
    classify_college,
    axis=1
)


# ============================================================
# SORT
# ============================================================

gender_df = gender_df.sort_values(
    by="final_predicted_2026_cutoff",
    ascending=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🎓 CollegeRank-AI")

st.subheader(
    "JEE Advanced Rank Based IIT CSE Recommendation & Cutoff Prediction System"
)

st.write(
    "Enter your JEE Advanced rank and gender category "
    "to explore IIT CSE colleges using estimated 2026 cutoffs."
)


# ============================================================
# STUDENT OVERVIEW
# ============================================================

st.divider()

st.header("📌 Student Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "JEE Advanced Rank",
        student_rank
    )


with col2:

    if gender == "Gender-Neutral":
        gender_display = "Gender-Neutral"
    else:
        gender_display = "Female"

    st.metric(
        "Gender",
        gender_display
    )


with col3:

    st.metric(
        "IITs Analyzed",
        len(gender_df)
    )


with col4:

    matches_found = len(
        gender_df[
            gender_df["category"] != "Outside Predicted Cutoff"
        ]
    )

    st.metric(
        "Matches Found",
        matches_found
    )


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

st.divider()

st.header("📊 Recommendation Summary")


strong_count = len(
    gender_df[
        gender_df["category"] == "Strong Match"
    ]
)

within_count = len(
    gender_df[
        gender_df["category"] == "Within Predicted Cutoff"
    ]
)

borderline_count = len(
    gender_df[
        gender_df["category"] == "Borderline"
    ]
)

outside_count = len(
    gender_df[
        gender_df["category"] == "Outside Predicted Cutoff"
    ]
)


s1, s2, s3, s4 = st.columns(4)


with s1:

    st.metric(
        "Strong Match",
        strong_count
    )


with s2:

    st.metric(
        "Within Cutoff",
        within_count
    )


with s3:

    st.metric(
        "Borderline",
        borderline_count
    )


with s4:

    st.metric(
        "Outside Cutoff",
        outside_count
    )


# ============================================================
# RECOMMENDED IITs
# ============================================================

st.divider()

st.header("🏫 IIT CSE Recommendations")

st.caption(
    "The following results are based on estimated 2026 closing ranks."
)


recommended_df = gender_df[
    gender_df["category"].isin(
        [
            "Strong Match",
            "Within Predicted Cutoff",
            "Borderline"
        ]
    )
].copy()


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if len(recommended_df) == 0:

    st.warning(
        "No IITs are currently within the selected "
        "recommendation range."
    )

else:

    for _, row in recommended_df.iterrows():

        institute = row["institute"]

        cutoff = int(
            row["final_predicted_2026_cutoff"]
        )

        latest_cutoff = int(
            row["latest_2025_cutoff"]
        )

        difference = int(
            row["rank_difference"]
        )

        category = row["category"]


        with st.container(border=True):

            st.subheader(
                f"🏛️ {institute}"
            )

            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Estimated 2026 Cutoff",
                    cutoff
                )


            with c2:

                st.metric(
                    "2025 Cutoff",
                    latest_cutoff
                )


            with c3:

                st.metric(
                    "Difference From Your Rank",
                    f"{difference:+d}"
                )


            if category == "Strong Match":

                st.success(
                    f"Recommendation: {category}"
                )

            elif category == "Within Predicted Cutoff":

                st.info(
                    f"Recommendation: {category}"
                )

            else:

                st.warning(
                    f"Recommendation: {category}"
                )


# ============================================================
# DETAILED TABLE
# ============================================================

st.divider()

st.header("📋 Detailed IIT Cutoff Table")


with st.expander(
    "View all IITs and prediction details"
):

    table_df = gender_df[
        [
            "institute",
            "latest_2025_cutoff",
            "predicted_2026_cutoff",
            "final_predicted_2026_cutoff",
            "rank_difference",
            "category"
        ]
    ].copy()


    table_df.columns = [
        "Institute",
        "2025 Cutoff",
        "Trend Prediction",
        "Final 2026 Estimate",
        "Difference",
        "Recommendation"
    ]


    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True
    )


# ============================================================
# PREDICTION METHODOLOGY
# ============================================================

st.divider()

st.header("⚙️ How CollegeRank-AI Works")


with st.expander(
    "Learn how the prediction system works"
):

    st.markdown(
        """
        ### Step 1 — Historical Data

        JoSAA IIT CSE cutoff data from 2021–2025
        is used for analysis.

        ### Step 2 — Data Preparation

        Historical data is separated into training,
        validation and testing periods.

        ### Step 3 — Historical Trend

        Historical cutoff trends are calculated
        separately for each IIT and gender category.

        ### Step 4 — 2026 Prediction

        The system combines the latest available
        cutoff with the historical trend estimate.

        ### Step 5 — Recommendation

        Your JEE Advanced rank is compared with
        the estimated 2026 closing rank.

        ### Important

        The 2026 cutoff values shown here are
        model-based estimates and are not guaranteed
        admission cutoffs.

        Actual JoSAA cutoffs can differ because of
        seat availability, student preferences,
        competition and other admission factors.
        """
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.header("📚 Project Information")


info1, info2 = st.columns(2)


with info1:

    st.markdown(
        """
        **Dataset**

        - JoSAA cutoff data
        - Years: 2021–2025
        - Institutes: IITs
        - Program: Computer Science & Engineering
        - Quota: AI
        - Seat Type: OPEN
        - Round: 2
        """
    )


with info2:

    st.markdown(
        """
        **Prediction System**

        - Historical trend analysis
        - Latest cutoff baseline
        - Hybrid 2026 estimate
        - Gender-specific prediction
        - Rank-based recommendation
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 CollegeRank-AI | JEE Advanced IIT CSE "
    "Prediction & Recommendation System"
)

st.caption(
    "Model predictions are estimates and should not "
    "be treated as guaranteed admission results."
)