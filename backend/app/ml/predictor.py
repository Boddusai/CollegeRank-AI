from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas.prediction import PredictionRequest, CollegePrediction


class JEEModelPredictor:
    """
    Phase-1 JEE/JoSAA predictor.

    Current supported dataset scope:
    - IIT
    - Computer Science and Engineering
    - Round 2
    - OPEN
    - AI quota
    - 2021-2025 historical data

    The trained XGBoost model predicts the closing rank.
    The student's rank is then compared with that predicted cutoff
    to generate a transparent recommendation score.
    """

    def __init__(self, model_dir: Path | None = None):
        # predictor.py:
        # CollegeRank-AI/backend/app/ml/predictor.py
        #
        # parents[3] = CollegeRank-AI
        self.project_root = Path(__file__).resolve().parents[3]

        # Use the verified final XGBoost model.
        self.model_path = self.project_root / "models" / "final_xgboost_model.pkl"

        # Use the verified JoSAA Phase-1 dataset.
        self.data_path = (
            self.project_root
            / "data"
            / "processed"
            / "josaa_iit_cse_open_2021_2025.csv"
        )

        self.available = self.model_path.exists()
        self.model = None

        if self.available:
            self.model = joblib.load(self.model_path)

    def predict(
        self,
        request: PredictionRequest,
        df: pd.DataFrame,
    ):
        if df.empty:
            return []

        if self.model is None:
            raise ValueError(
                "Final XGBoost model is not available."
            )

        # ---------------------------------------------------------
        # Current Phase-1 dataset constraints
        # ---------------------------------------------------------
        candidates = df.copy()

        # The current dataset contains only IIT records.
        if "institute_type" in candidates.columns:
            candidates = candidates[
                candidates["institute_type"].astype(str).str.upper() == "IIT"
            ]

        # Current Phase-1 dataset is CSE only.
        if "academic_program" in candidates.columns:
            candidates = candidates[
                candidates["academic_program"]
                .astype(str)
                .str.contains(
                    "Computer Science and Engineering",
                    case=False,
                    na=False,
                )
            ]

        # Current Phase-1 dataset is Round 2.
        if "round" in candidates.columns:
            candidates = candidates[candidates["round"] == 2]

        # Current Phase-1 dataset is OPEN.
        if "seat_type" in candidates.columns:
            candidates = candidates[
                candidates["seat_type"].astype(str).str.upper() == "OPEN"
            ]

        # Current Phase-1 dataset is All India quota.
        if "quota" in candidates.columns:
            candidates = candidates[
                candidates["quota"].astype(str).str.upper() == "AI"
            ]

        # ---------------------------------------------------------
        # Gender filtering
        # ---------------------------------------------------------
        requested_gender = self._normalize_gender(request.gender)

        if requested_gender:
            candidates = candidates[
                candidates["gender"]
                .apply(self._normalize_gender)
                .eq(requested_gender)
            ]

        # ---------------------------------------------------------
        # Branch filtering
        # ---------------------------------------------------------
        if request.preferred_branch:
            branch_text = request.preferred_branch.strip().lower()

            # The current Phase-1 dataset only has CSE.
            if branch_text not in {
                "computer science",
                "computer science and engineering",
                "cse",
            }:
                return []

        if candidates.empty:
            return []

        # ---------------------------------------------------------
        # Historical years
        # ---------------------------------------------------------
        latest_year = int(candidates["year"].max())

        historical_2025 = candidates[
            candidates["year"] == latest_year
        ].copy()

        historical_2024 = candidates[
            candidates["year"] == latest_year - 1
        ].copy()

        # ---------------------------------------------------------
        # Generate 2026 prediction rows
        #
        # The model was trained on:
        #   2021-2023
        #
        # We preserve the model evaluation model and use year=2026
        # only for inference.
        # ---------------------------------------------------------
        prediction_rows = historical_2025.copy()
        prediction_rows["year"] = 2026

        feature_columns = [
            "year",
            "round",
            "institute_type",
            "institute",
            "academic_program",
            "quota",
            "seat_type",
            "gender",
        ]

        prediction_input = prediction_rows[feature_columns].copy()

        predicted_cutoffs = self.model.predict(prediction_input)

        prediction_rows["predicted_cutoff"] = np.maximum(
            1,
            np.rint(predicted_cutoffs),
        ).astype(int)

        # ---------------------------------------------------------
        # Build result objects
        # ---------------------------------------------------------
        output = []

        for _, row in prediction_rows.iterrows():

            college = str(row["institute"])
            branch = str(row["academic_program"])

            predicted_cutoff = int(row["predicted_cutoff"])

            # Latest historical cutoff.
            latest_record = historical_2025[
                historical_2025["institute"] == college
            ]

            historical_cutoff = None

            if not latest_record.empty:
                value = latest_record.iloc[-1]["closing_rank"]

                if pd.notna(value):
                    historical_cutoff = float(value)

            # Previous-year cutoff.
            previous_record = historical_2024[
                historical_2024["institute"] == college
            ]

            previous_year_cutoff = None

            if not previous_record.empty:
                value = previous_record.iloc[-1]["closing_rank"]

                if pd.notna(value):
                    previous_year_cutoff = float(value)

            # -----------------------------------------------------
            # Recommendation score
            # -----------------------------------------------------
            score = self._match_score(
                student_rank=request.rank,
                predicted_cutoff=predicted_cutoff,
            )

            # -----------------------------------------------------
            # Historical trend
            # -----------------------------------------------------
            trend = self._trend(
                historical_cutoff,
                previous_year_cutoff,
                predicted_cutoff,
            )

            # -----------------------------------------------------
            # Explanation
            # -----------------------------------------------------
            explanation = self._explanation(
                rank=request.rank,
                predicted_cutoff=predicted_cutoff,
                historical_cutoff=historical_cutoff,
                previous_year_cutoff=previous_year_cutoff,
                score=score,
            )

            output.append(
                CollegePrediction(
                    college=college,
                    branch=branch,
                    prediction_score=round(score, 3),
                    predicted_cutoff=float(predicted_cutoff),
                    historical_cutoff=historical_cutoff,
                    previous_year_cutoff=previous_year_cutoff,
                    trend=trend,
                    explanation=explanation,
                )
            )

        # Highest match score first.
        output.sort(
            key=lambda item: item.prediction_score,
            reverse=True,
        )

        return output[:10]

    # =============================================================
    # Helper methods
    # =============================================================

    @staticmethod
    def _normalize_gender(gender):
        """
        Convert UI gender values into the dataset's two categories.
        """

        if gender is None:
            return None

        value = str(gender).strip().lower()

        if "female" in value:
            return "female"

        if "gender-neutral" in value:
            return "gender-neutral"

        if value in {"neutral", "gender neutral", "male"}:
            return "gender-neutral"

        return value

    @staticmethod
    def _match_score(student_rank, predicted_cutoff):
        """
        Convert the rank-vs-cutoff relationship into a transparent
        recommendation score.

        IMPORTANT:
        This is NOT an XGBoost probability.

        XGBoost predicts the closing rank.
        This score measures how comfortably the student's rank
        compares with that predicted closing rank.
        """

        if student_rank is None:
            return 0.5

        student_rank = max(float(student_rank), 1.0)
        predicted_cutoff = max(float(predicted_cutoff), 1.0)

        # A rank lower than the predicted closing rank is favorable.
        ratio = predicted_cutoff / student_rank

        # Smooth score centered around the predicted cutoff.
        score = 1.0 / (
            1.0 + np.exp(-3.0 * (ratio - 1.0))
        )

        return float(np.clip(score, 0.01, 0.99))

    @staticmethod
    def _trend(
        historical_cutoff,
        previous_year_cutoff,
        predicted_cutoff,
    ):
        """
        Describe the movement from previous year -> latest historical
        year -> ML predicted year.
        """

        if (
            previous_year_cutoff is None
            or historical_cutoff is None
        ):
            return "Insufficient historical data"

        if (
            predicted_cutoff > historical_cutoff
            and historical_cutoff > previous_year_cutoff
        ):
            return "Closing rank trending higher"

        if (
            predicted_cutoff < historical_cutoff
            and historical_cutoff < previous_year_cutoff
        ):
            return "Closing rank trending lower"

        if predicted_cutoff > historical_cutoff:
            return "Predicted closing rank may increase"

        if predicted_cutoff < historical_cutoff:
            return "Predicted closing rank may decrease"

        return "Stable"

    @staticmethod
    def _explanation(
        rank,
        predicted_cutoff,
        historical_cutoff,
        previous_year_cutoff,
        score,
    ):
        if rank is None:
            return (
                f"ML-predicted 2026 closing rank is "
                f"{predicted_cutoff:,}. "
                "Enter your rank to compare it with the "
                "predicted closing rank."
            )

        if rank <= predicted_cutoff:
            position = "within or better than"
        else:
            position = "beyond"

        historical_text = ""

        if historical_cutoff is not None:
            historical_text = (
                f" The latest historical closing rank was "
                f"{historical_cutoff:,.0f}."
            )

        previous_text = ""

        if previous_year_cutoff is not None:
            previous_text = (
                f" The previous-year closing rank was "
                f"{previous_year_cutoff:,.0f}."
            )

        
        return (
            f"Student rank {int(rank):,} is {position} the "
            f"ML-predicted 2026 closing rank of "
            f"{predicted_cutoff:,}. "
            "Compare your rank with this estimate cautiously; "
            "the prediction has uncertainty and does not "
            "represent an admission probability."
            f"{historical_text}"
            f"{previous_text} "
            "This is an ML-based historical estimate, "
            "not a guaranteed admission result."
        )
