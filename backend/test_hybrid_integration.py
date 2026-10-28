
import numpy as np
import pandas as pd

from app.services import prediction_service
from unittest.mock import Mock


def test_hybrid_helper_is_used_for_jee_main(monkeypatch):
    service = prediction_service.PredictionService()

    # Avoid loading real files or relying on the trained model.
    service.main_model = Mock()

    historical = pd.DataFrame([
        {
            "exam": "JEE Main",
            "year": 2025,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "opening_rank": 1,
            "closing_rank": 100,
        }
    ])

    monkeypatch.setattr(
        prediction_service.pd,
        "read_csv",
        lambda path: historical.copy(),
    )

    service.main_model.predict.return_value = np.array([120.0])

    calls = {}

    def fake_hybrid_predictor(
        prediction_rows,
        historical_data,
        rf_predictions,
        rf_weight=0.75,
    ):
        calls["called"] = True
        calls["rf_weight"] = rf_weight
        calls["rf_predictions"] = list(rf_predictions)
        return np.array([110])

    monkeypatch.setattr(
        prediction_service,
        "calculate_hybrid_cutoffs",
        fake_hybrid_predictor,
    )

    # Keep the fixture's institute mapped to an unrelated state only
    # if needed; AI quota does not require state matching.
    request = prediction_service.PredictionRequest(
        exam="JEE Main",
        rank=100,
        category="OPEN",
        gender="Gender-Neutral",
        state="Telangana",
        preferred_branch="CSE",
        round=2,
        quota="AI",
    )

    response = service._predict_jee_main(request)

    assert calls["called"] is True
    assert calls["rf_weight"] == 0.75
    assert calls["rf_predictions"] == [120.0]
    assert response.model_available is True
    assert len(response.predictions) == 1
    assert response.predictions[0].predicted_cutoff == 110.0


def test_hybrid_helper_excludes_prediction_year_and_future_years():
    from app.services.hybrid_predictor import calculate_hybrid_cutoffs

    prediction_rows = pd.DataFrame([
        {
            "exam": "JEE Main",
            "year": 2026,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
        }
    ])

    historical = pd.DataFrame([
        {
            "exam": "JEE Main",
            "year": 2024,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 100,
        },
        {
            "exam": "JEE Main",
            "year": 2025,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 200,
        },
        {
            "exam": "JEE Main",
            "year": 2026,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 99999,
        },
        {
            "exam": "JEE Main",
            "year": 2027,
            "round": 2,
            "institute_type": "NIT",
            "institute": "Test NIT",
            "academic_program": "Computer Science and Engineering",
            "quota": "AI",
            "seat_type": "OPEN",
            "gender": "Gender-Neutral",
            "closing_rank": 88888,
        },
    ])

    result = calculate_hybrid_cutoffs(
        prediction_rows=prediction_rows,
        historical_data=historical,
        rf_predictions=[1000],
        rf_weight=0.75,
    )

    # Eligible years are 2024 and 2025 only.
    # Their weighted historical cutoff is (100*1 + 200*2)/3.
    # Hybrid = 75% * 1000 + 25% * (500/3), rounded to 792.
    assert result.tolist() == [792]

def test_jee_advanced_does_not_call_hybrid_helper(monkeypatch):
    service = prediction_service.PredictionService()

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Hybrid helper must not be called for JEE Advanced"
        )

    monkeypatch.setattr(
        prediction_service,
        "calculate_hybrid_cutoffs",
        fail_if_called,
    )

    request = prediction_service.PredictionRequest(
        exam="JEE Advanced",
        rank=1000,
        category="OPEN",
        gender="Gender-Neutral",
        state="Telangana",
        preferred_branch="CSE",
        round=2,
        quota="AI",
    )

    # If the Advanced model/data are unavailable, the service should
    # return its existing unavailable response without using the helper.
    response = service._predict_jee_advanced(request)

    assert response.exam == "JEE Advanced"
    assert isinstance(response.predictions, list)
