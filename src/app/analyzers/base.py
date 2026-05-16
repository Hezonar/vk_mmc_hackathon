from typing import Protocol

from app.domain.patient import PatientRecord
from app.domain.prediction import PredictionResult


class Analyzer(Protocol):
    id: str
    version: str

    def analyze(self, patient: PatientRecord) -> PredictionResult: ...
