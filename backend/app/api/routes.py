from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()


@router.get("/exams")
def exams():
    return {
        "exams": [
            "JEE Main", "JEE Advanced", "NEET UG", "TG EAPCET",
            "AP EAPCET", "MHT CET", "KCET", "COMEDK UGET",
            "BITSAT", "WBJEE"
        ]
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return service.predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
