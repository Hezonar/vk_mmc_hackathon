import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGITAL_APP = ROOT / "digital_propathologist_mvp"
if str(DIGITAL_APP) not in sys.path:
    sys.path.insert(0, str(DIGITAL_APP))

from digital_propathologist.recommendations import detect_research_warnings
from digital_propathologist.review_state import load_review_state, mark_reviewed, reviewed_count
from digital_propathologist.types import Exam, PredictionResult, SpecialistConclusion
from digital_propathologist.ui_html import render_exam_details
from digital_propathologist.ml_model import predict_exam_with_candidate_model


def test_empty_research_name_does_not_create_warning():
    conclusions = [
        SpecialistConclusion(specialist="Спирометрия"),
        SpecialistConclusion(specialist="Электрокардиография"),
    ]

    assert detect_research_warnings(conclusions) == []


def test_non_empty_medical_result_can_create_warning():
    conclusions = [
        SpecialistConclusion(
            specialist="Врач Терапевт",
            health_group="2 группа",
            mkb_code="I10",
            mkb_description="Эссенциальная гипертензия",
        )
    ]

    warnings = detect_research_warnings(conclusions)

    assert len(warnings) == 1
    assert warnings[0].kind == "артериальное давление"


def test_warning_section_is_hidden_without_warnings():
    exam = Exam(
        exam_row_id="1016231267",
        patient_id="310942",
        assigned_harmful_factors=["4.4"],
        specialist_conclusions=[
            SpecialistConclusion(specialist="Спирометрия"),
            SpecialistConclusion(specialist="Электрокардиография"),
        ],
    )

    html = render_exam_details(exam, PredictionResult())

    assert "<h2>Настораживающие признаки</h2>" not in html


def test_review_state_counts_once_and_rolls_day(tmp_path):
    path = tmp_path / "review_state.json"

    state, created = mark_reviewed("310942", "1016231267", path=path, day="2026-05-17")
    assert created is True
    assert reviewed_count(state) == 1

    state, created = mark_reviewed("310942", "1016231267", path=path, day="2026-05-17")
    assert created is False
    assert reviewed_count(state) == 1

    state = load_review_state(path, day="2026-05-18")
    assert reviewed_count(state) == 0
    assert len(state["history"]) == 1


def test_ml_result_contains_verdict_model_and_scores_for_insufficient_data():
    exam = Exam(
        exam_row_id="1016231267",
        patient_id="310942",
        assigned_harmful_factors=[],
        specialist_conclusions=[],
    )

    result = predict_exam_with_candidate_model(exam)

    assert result.ml_verdict == "insufficient"
    assert result.model_version
    assert result.factor_scores == []


def test_render_ml_footer_uses_fit_language():
    exam = Exam(
        exam_row_id="1016231267",
        patient_id="310942",
        assigned_harmful_factors=["4.4"],
        specialist_conclusions=[
            SpecialistConclusion(specialist="Врач-оториноларинголог", conclusion="Слух в пределах нормы.")
        ],
    )
    prediction = PredictionResult(
        done=True,
        status="ok",
        ml_verdict="fit",
        model_version="006_candidate_factor_binary",
        threshold=0.42,
        factor_scores=[{"factor": "4.4", "score": 0.12, "status": "fit"}],
    )

    html = render_exam_details(exam, prediction)

    assert "Годен по оценке ML" in html
    assert "006_candidate_factor_binary" in html
