import app.analyzers.example_rules  # noqa: F401
import pytest

from app.analyzers.registry import resolve
from app.domain.errors import AnalyzerExecutionError, AnalyzerNotFoundError
from app.domain.patient import PatientRecord
from app.domain.prediction import HealthLabel, PredictionResult
from app.services.prediction import PredictionService


def test_resolve_unknown():
    with pytest.raises(AnalyzerNotFoundError):
        resolve("does-not-exist")


class _Broken:
    id = "broken"
    version = "0.0.1"

    def analyze(self, patient: PatientRecord) -> PredictionResult:
        raise RuntimeError("fail")


def test_service_wraps_analyzer_errors():
    svc = PredictionService()
    with pytest.raises(AnalyzerExecutionError):
        svc._run_analyzer(_Broken(), PatientRecord(id="x"))


def test_service_predict_default():
    svc = PredictionService()
    out = svc.predict(PatientRecord(id="z", age=30))
    assert out.label == HealthLabel.healthy
