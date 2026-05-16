import app.analyzers.example_rules  # noqa: F401
from app.analyzers.example_rules import DemoRulesAnalyzer
from app.domain.patient import PatientRecord
from app.domain.prediction import HealthLabel


def test_demo_analyzer_young_is_healthy():
    a = DemoRulesAnalyzer()
    r = a.analyze(PatientRecord(id="1", age=40))
    assert r.label == HealthLabel.healthy
    assert 0.5 <= r.confidence <= 1.0


def test_demo_analyzer_old_is_unhealthy():
    a = DemoRulesAnalyzer()
    r = a.analyze(PatientRecord(id="1", age=70))
    assert r.label == HealthLabel.unhealthy


def test_demo_analyzer_missing_age_neutral():
    a = DemoRulesAnalyzer()
    r = a.analyze(PatientRecord(id="1"))
    assert r.label == HealthLabel.healthy
    assert r.confidence < 0.6
