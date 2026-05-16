from enum import Enum

from pydantic import BaseModel, Field


class HealthLabel(str, Enum):
    healthy = "healthy"
    unhealthy = "unhealthy"


class PredictionResult(BaseModel):
    label: HealthLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    analyzer_id: str
    analyzer_version: str
    details: str | None = None
