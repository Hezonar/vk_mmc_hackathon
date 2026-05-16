from app.analyzers.registry import register
from app.domain.patient import PatientRecord
from app.domain.prediction import HealthLabel, PredictionResult


class DemoRulesAnalyzer:
    id = "demo_rules"
    version = "1.0.0"

    def analyze(self, patient: PatientRecord) -> PredictionResult:
        age = patient.age
        if age is None:
            return PredictionResult(
                label=HealthLabel.healthy,
                confidence=0.55,
                analyzer_id=self.id,
                analyzer_version=self.version,
                details="demo_risk_score: age missing, neutral baseline",
            )
        threshold = 65.0
        distance = abs(float(age) - threshold)
        confidence = min(0.5 + distance / 40.0, 0.98)
        if age >= int(threshold):
            return PredictionResult(
                label=HealthLabel.unhealthy,
                confidence=confidence,
                analyzer_id=self.id,
                analyzer_version=self.version,
                details="demo_risk_score: age at or above demo threshold",
            )
        return PredictionResult(
            label=HealthLabel.healthy,
            confidence=confidence,
            analyzer_id=self.id,
            analyzer_version=self.version,
            details="demo_risk_score: age below demo threshold",
        )


_demo = DemoRulesAnalyzer()
register(_demo)
