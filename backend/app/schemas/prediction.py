from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    exam: str = "JEE Main"
    rank: Optional[int] = Field(default=None, ge=1)
    score: Optional[float] = Field(default=None, ge=0)
    percentile: Optional[float] = Field(default=None, ge=0, le=100)
    category: str
    gender: str
    state: str
    preferred_branch: Optional[str] = None
    round: Optional[int] = Field(default=None, ge=1)
    quota: Optional[str] = None


class CollegePrediction(BaseModel):
    college: str
    branch: str
    prediction_score: float
    historical_cutoff: Optional[float] = None
    previous_year_cutoff: Optional[float] = None
    trend: str
    explanation: str
    predicted_cutoff: Optional[float] = None


class PredictionResponse(BaseModel):
    exam: str
    model_available: bool
    message: str
    predictions: list[CollegePrediction]
