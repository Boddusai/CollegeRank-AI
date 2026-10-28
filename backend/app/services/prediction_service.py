
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.ml.predictor import JEEModelPredictor
from app.services.hybrid_predictor import calculate_hybrid_cutoffs

class PredictionService:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[3]

        # Existing JEE Advanced files
        self.advanced_data_path = (
            self.root / "data" / "processed"
            / "josaa_iit_cse_open_2021_2025.csv"
        )
        self.advanced_predictor = JEEModelPredictor()

        # JEE Main files
        self.main_data_path = (
            self.root / "data" / "processed"
            / "jee_main_cse_round2_model_data.csv"
        )
        self.main_model_path = (
            self.root / "models"
            / "jee_main_cse_round2_random_forest.pkl"
        )

        self.main_model = None
        if self.main_model_path.exists():
            self.main_model = joblib.load(self.main_model_path)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        exam = request.exam.strip().lower()

        # The current collected datasets support Round 2 only.
        if request.round is not None and request.round != 2:
            return PredictionResponse(
                exam=request.exam,
                model_available=False,
                message=(
                    "The current dataset supports counselling Round 2 only. "
                    "Other rounds are not available until verified data "
                    "and a tested prediction model are added."
                ),
                predictions=[],
            )

        if exam == "jee main":
            return self._predict_jee_main(request)

        if exam == "jee advanced":
            return self._predict_jee_advanced(request)

        return PredictionResponse(
            exam=request.exam,
            model_available=False,
            message=(
                "This exam is not implemented yet. Currently supported: "
                "JEE Main and JEE Advanced. Other exams need verified data "
                "and a tested prediction method."
            ),
            predictions=[],
        )

    def _predict_jee_advanced(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        if not self.advanced_predictor.available:
            return PredictionResponse(
                exam=request.exam,
                model_available=False,
                message="The JEE Advanced model is unavailable.",
                predictions=[],
            )

        if not self.advanced_data_path.exists():
            return PredictionResponse(
                exam=request.exam,
                model_available=False,
                message="The JEE Advanced historical dataset is unavailable.",
                predictions=[],
            )

        df = pd.read_csv(self.advanced_data_path)
        results = self.advanced_predictor.predict(request, df)

        return PredictionResponse(
            exam=request.exam,
            model_available=True,
            message=(
                "JEE Advanced historical ML estimate. "
                "Not a guaranteed admission result or probability."
            ),
            predictions=results,
        )

    def _predict_jee_main(
        self, request: PredictionRequest
    ) -> PredictionResponse:
        if self.main_model is None or not self.main_data_path.exists():
            return PredictionResponse(
                exam=request.exam,
                model_available=False,
                message="The JEE Main model or dataset is unavailable.",
                predictions=[],
            )

        if request.rank is None:
            return PredictionResponse(
                exam=request.exam,
                model_available=True,
                message="Enter your JEE Main rank to compare it with cutoffs.",
                predictions=[],
            )

        df = pd.read_csv(self.main_data_path)

        # Use the latest available historical year as the prediction base.
        latest_year = int(df["year"].max())
        latest = df[df["year"] == latest_year].copy()

        # Apply the selected JoSAA quota.
        selected_quota = (getattr(request, "quota", None) or "").strip().upper()
        student_state = self._normalize_state(request.state)

        if selected_quota == "AI":
            latest = latest[
                latest["quota"].astype(str).str.strip().str.upper() == "AI"
            ]

        elif selected_quota in {"HS", "OS"}:
            institute_states = latest["institute"].apply(
                self._get_institute_state
            )

            if selected_quota == "HS":
                latest = latest[
                    (latest["quota"].astype(str).str.strip().str.upper() == "HS")
                    & (institute_states == student_state)
                ]
            else:
                latest = latest[
                    (latest["quota"].astype(str).str.strip().str.upper() == "OS")
                    & (institute_states.notna())
                    & (institute_states != student_state)
                ]

        # Do not pretend that HS alone identifies the student's home state.
        # State matching requires a verified institute-to-state mapping.


        # Apply the requested category.
        category = self._normalize(request.category)
        latest = latest[
            latest["seat_type"].apply(self._normalize) == category
        ]

        # Match the requested gender to the dataset's gender categories.
        requested_gender = self._normalize(request.gender)
        if requested_gender in {"female", "female-only"}:
            latest = latest[
                latest["gender"].apply(self._normalize).str.contains(
                    "female", na=False
                )
            ]
        else:
            latest = latest[
                latest["gender"].apply(self._normalize).str.contains(
                    "gender-neutral", na=False
                )
            ]

        # The current collected dataset contains CSE-related programs.
        if request.preferred_branch:
            branch = request.preferred_branch.strip().lower()
            if branch not in {
                "cse", "computer science",
                "computer science and engineering",
            }:
                latest = latest[
                    latest["academic_program"].str.contains(
                        request.preferred_branch, case=False, na=False
                    )
                ]

        if latest.empty:
            return PredictionResponse(
                exam=request.exam,
                model_available=True,
                message=(
                    "No matching records were found for the selected "
                    "category, gender, and branch in the collected dataset."
                ),
                predictions=[],
            )

        # Predict next-year closing ranks using the model's required features.
        prediction_rows = latest.copy()
        prediction_rows["year"] = latest_year + 1

        feature_columns = [
            "exam", "institute_type", "institute", "academic_program",
            "quota", "seat_type", "gender", "year", "round", "opening_rank",
        ]

        # Generate the existing Random Forest predictions.
        rf_predictions = self.main_model.predict(
            prediction_rows[feature_columns]
        )

        # Combine RF predictions with eligible historical cutoffs.
        # The helper excludes records from the prediction year and future years.
        predicted = calculate_hybrid_cutoffs(
            prediction_rows=prediction_rows,
            historical_data=df,
            rf_predictions=rf_predictions,
            rf_weight=0.75,
        )

        prediction_rows["predicted_cutoff"] = predicted

        output = []
        for _, row in prediction_rows.iterrows():
            cutoff = int(row["predicted_cutoff"])
            student_rank = int(request.rank)

            # This is a comparison score, not an admission probability.
            if student_rank <= cutoff:
                score = round(student_rank / cutoff, 3)
            else:
                score = round(cutoff / student_rank, 3)

            historical = float(row["closing_rank"])
            program = str(row["academic_program"])
            quota = str(row["quota"])

            # Find the matching record from the previous year.
            previous_year = latest_year - 1

            previous_rows = df[
                (df["year"] == previous_year)
                & (df["institute"].astype(str) == str(row["institute"]))
                & (
                    df["academic_program"].astype(str)
                    == str(row["academic_program"])
                )
                & (df["quota"].astype(str) == str(row["quota"]))
                & (df["seat_type"].astype(str) == str(row["seat_type"]))
                & (df["gender"].astype(str) == str(row["gender"]))
                & (df["round"] == row["round"])
            ]

            previous_year_cutoff = None

            if not previous_rows.empty:
                previous_year_cutoff = float(
                    previous_rows.iloc[0]["closing_rank"]
                )

            output.append({
                "college": str(row["institute"]),
                "branch": program,
                "prediction_score": round(score, 3),
                "historical_cutoff": historical,
                "previous_year_cutoff": previous_year_cutoff,
                "trend": (
                    f"Historical closing rank changed from "
                    f"{previous_year_cutoff:,.0f} to {historical:,.0f}; "
                    f"model-estimated next-year closing rank is {cutoff:,}."
                    if previous_year_cutoff is not None
                    else self._describe_trend(historical, cutoff)
                ),
                "explanation": (
                    f"Quota: {quota}. Latest historical closing rank: "
                    f"{historical:,.0f}. Model-estimated next-year closing "
                    f"rank: {cutoff:,}. This is a historical estimate, "
                    "not an admission probability or guarantee."
                ),
                "predicted_cutoff": float(cutoff),
            })

        output.sort(key=lambda item: item["prediction_score"], reverse=True)

        return PredictionResponse(
            exam=request.exam,
            model_available=True,
            message=(
                f"Estimates based on collected {latest_year} JoSAA Round 2 "
                "records. The dataset is limited to collected CSE-related "
                "NIT, IIIT and GFTI records. "
                "Quota filtering uses the selected JoSAA quota. "
                "Home State and Other State filtering uses the configured "
                "institute-state mapping. Results depend on the accuracy "
                "and completeness of that mapping. "
                "Scores are comparison indicators, not admission probabilities."
            ),
            predictions=output[:10],
        )

    @staticmethod
    def _normalize_state(value):
        value = " ".join(str(value).strip().lower().replace("_", " ").split())

        aliases = {
            "telengana": "telangana",
            "orissa": "odisha",
            "pondicherry": "puducherry",
            "nct of delhi": "delhi",
            "uttaranchal": "uttarakhand",
        }

        return aliases.get(value, value)

    @staticmethod
    def _get_institute_state(institute):
        name = " ".join(str(institute).lower().split())

        institute_states = {
            "dr b r ambedkar national institute of technology, jalandhar": "punjab",
            "malaviya national institute of technology jaipur": "rajasthan",
            "maulana azad national institute of technology bhopal": "madhya pradesh",
            "motilal nehru national institute of technology allahabad": "uttar pradesh",
            "national institute of technology  agartala": "tripura",
            "national institute of technology agartala": "tripura",
            "national institute of technology calicut": "kerala",
            "national institute of technology delhi": "delhi",
            "national institute of technology durgapur": "west bengal",
            "national institute of technology goa": "goa",
            "national institute of technology hamirpur": "himachal pradesh",
            "national institute of technology karnataka, surathkal": "karnataka",
            "national institute of technology meghalaya": "meghalaya",
            "national institute of technology nagaland": "nagaland",
            "national institute of technology patna": "bihar",
            "national institute of technology puducherry": "puducherry",
            "national institute of technology raipur": "chhattisgarh",
            "national institute of technology sikkim": "sikkim",
            "national institute of technology arunachal pradesh": "arunachal pradesh",
            "national institute of technology, jamshedpur": "jharkhand",
            "national institute of technology, man ipur": "manipur",
            "national institute of technology, mizoram": "mizoram",
            "national institute of technology, rourkela": "odisha",
            "national institute of technology, silchar": "assam",
            "national institute of technology, tiruchirappalli": "tamil nadu",
            "national institute of technology, uttarakhand": "uttarakhand",
            "national institute of technology, warangal": "telangana",
            "sardar vallabhbhai national institute of technology, surat": "gujarat",
            "visvesvaraya national institute of technology, nagpur": "maharashtra",
            "national institute of technology, andhra pradesh": "andhra pradesh",
            "indian institute of engineering science and technology, shibpur": "west bengal",
            "assam university, silchar": "assam",
            "birla institute of technology, mesra, ranchi": "jharkhand",
            "punjab engineering college, chandigarh": "chandigarh",
            "pondicherry engineering college, puducherry": "puducherry",
            "puducherry technological university, puducherry": "puducherry",
            "birla institute of technology, deoghar off-campus": "jharkhand",
            "birla institute of technology, patna off-campus": "bihar",
            "national institute of technology, kurukshetra": "haryana",
        }

        # Handle punctuation and repeated spaces consistently.
        return institute_states.get(name)

    @staticmethod
    def _normalize(value):
        value = str(value).strip().lower()
        value = value.replace("_", "-")
        value = " ".join(value.split())

        aliases = {
            "open": "open",
            "general": "open",
            "open (pwd)": "open (pwd)",
            "ews": "ews",
            "obc": "obc-ncl",
            "obc-ncl": "obc-ncl",
            "sc": "sc",
            "st": "st",
            "gender neutral": "gender-neutral",
            "gender-neutral": "gender-neutral",
            "female": "female-only",
            "female-only": "female-only",
        }
        return aliases.get(value, value)

    @staticmethod
    def _describe_trend(historical, predicted):
        if predicted > historical:
            return "Predicted closing rank may increase"
        if predicted < historical:
            return "Predicted closing rank may decrease"
        return "Stable"
