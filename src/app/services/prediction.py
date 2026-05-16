from app.analyzers import resolve
from app.analyzers.base import Analyzer
from app.domain.errors import AnalyzerExecutionError
from app.domain.patient import PatientRecord
from app.domain.prediction import PredictionResult


class PredictionService:
    def predict(
        self,
        patient: PatientRecord,
        analyzer_id: str | None = None,
    ) -> PredictionResult:
        analyzer = resolve(analyzer_id)
        return self._run_analyzer(analyzer, patient)

    def _run_analyzer(self, analyzer: Analyzer, patient: PatientRecord) -> PredictionResult:
        try:
            return analyzer.analyze(patient)
        except AnalyzerExecutionError:
            raise
        except Exception as exc:
            raise AnalyzerExecutionError(
                analyzer.id,
                f"Analyzer failed: {exc}",
            ) from exc
