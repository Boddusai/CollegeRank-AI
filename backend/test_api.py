
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_exam_list_contains_all_ten_exams():
    response = client.get("/api/exams")
    assert response.status_code == 200

    exams = response.json()["exams"]
    assert len(exams) == 10
    assert "JEE Main" in exams
    assert "JEE Advanced" in exams
    assert "NEET UG" in exams


def test_rank_zero_is_rejected():
    response = client.post(
        "/api/predict",
        json={
            "exam": "JEE Main",
            "rank": 0,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "state": "Telangana",
            "preferred_branch": "CSE",
            "round": 2,
            "quota": "AI",
        },
    )
    assert response.status_code == 422


def test_unsupported_exam_returns_clear_message():
    response = client.post(
        "/api/predict",
        json={
            "exam": "NEET UG",
            "rank": 1000,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "state": "Telangana",
            "preferred_branch": "CSE",
            "round": 2,
            "quota": "AI",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_available"] is False
    assert data["predictions"] == []
    assert "not implemented yet" in data["message"].lower()


def test_unsupported_counselling_round_is_not_silently_accepted():
    response = client.post(
        "/api/predict",
        json={
            "exam": "JEE Main",
            "rank": 15000,
            "category": "OPEN",
            "gender": "Gender-Neutral",
            "state": "Telangana",
            "preferred_branch": "CSE",
            "round": 1,
            "quota": "AI",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model_available"] is False
    assert data["predictions"] == []
    assert "round 2 only" in data["message"].lower()
