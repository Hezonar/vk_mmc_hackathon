from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.domain.patient import PatientRecord
from app.domain.prediction import PredictionResult
from app.services.prediction import PredictionService

router = APIRouter(prefix="/v1", tags=["api"])


class PredictRequest(BaseModel):
    patient: PatientRecord
    analyzer_id: str | None = Field(default=None)


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


@router.post("/predict", response_model=PredictionResult)
def predict_v1(
    body: PredictRequest,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> PredictionResult:
    return service.predict(body.patient, analyzer_id=body.analyzer_id)
