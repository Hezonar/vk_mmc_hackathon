from __future__ import annotations

from .highlighting import build_highlight_links, card_has_warning
from .types import Exam, PredictionResult


def _heuristic_factors(exam: Exam) -> list[str]:
    """Fallback only for demo when the row says there are contraindications but no factors are given."""
    links = build_highlight_links(exam)
    result: list[str] = []
    for factor in exam.assigned_harmful_factors:
        linked = links.get(factor, [])
        if any(0 <= idx < len(exam.specialist_conclusions) and card_has_warning(exam.specialist_conclusions[idx]) for idx in linked):
            result.append(factor)
    return result


def predict_exam(exam: Exam) -> PredictionResult:
    """
    Demo ML stub.

    Replace this function with a real model call later. The interface is intentionally stable:
    input: parsed Exam
    output: PredictionResult
    """
    # Случай 1 из презентации заказчика: медицинских данных недостаточно.
    # Для MVP считаем критически недостаточными две ситуации:
    # 1) нет факторов вредности/видов работ из направления;
    # 2) нет распарсенных заключений профильных врачей/исследований.
    empty_conclusions = [item for item in exam.specialist_conclusions if not item.has_meaningful_result]
    complete_conclusions = [item for item in exam.specialist_conclusions if item.has_meaningful_result]
    if not exam.assigned_harmful_factors or not exam.specialist_conclusions or empty_conclusions:
        missing = []
        if not exam.assigned_harmful_factors:
            missing.append("факторы вредности/виды работ")
        if not exam.specialist_conclusions:
            missing.append("заключения профильных врачей и результаты исследований")
        if empty_conclusions:
            missing.append(
                f"заполненные заключения/результаты исследований: {len(complete_conclusions)} из {len(exam.specialist_conclusions)}"
            )
        return PredictionResult(
            done=True,
            factors=[],
            status="insufficient",
            explanation=(
                "Недостаточно медицинских данных для вынесения предварительного вывода. "
                "Не переданы: " + ", ".join(missing) + ". "
                "Профосмотр должен быть приостановлен до получения дополнительных данных."
            ),
            used_stub=True,
            linked_factors={},
        )

    if exam.contraindicated_factors:
        factors = [f for f in exam.contraindicated_factors if f in exam.assigned_harmful_factors] or exam.contraindicated_factors
        status = "risk" if factors else "ok"
        explanation = (
            "Демонстрационная заглушка использовала поле contraindicated_factors из CSV. "
            "В реальном продукте здесь будет вызов ML-модели или API."
        )
    elif exam.has_contraindications is True:
        factors = _heuristic_factors(exam)
        status = "risk" if factors else "insufficient"
        explanation = (
            "В строке указано наличие противопоказаний, но конкретные факторы не заполнены. "
            "Заглушка попробовала выделить факторы по текстам заключений."
        )
    else:
        factors = []
        status = "ok"
        explanation = "Демонстрационная заглушка не выделила противопоказанных факторов для данного осмотра."

    links = build_highlight_links(exam, PredictionResult(done=True, factors=factors, status=status))
    return PredictionResult(
        done=True,
        factors=factors,
        status=status,
        explanation=explanation,
        used_stub=True,
        linked_factors=links,
    )
