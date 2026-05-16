import re
from typing import Any

import pandas as pd

from app.profpath.constants import FACTOR_HINTS, RISK_TERMS
from app.profpath.parsing import get_parsed_info, parse_bool, safe_str, split_codes


def build_model_input(row: pd.Series, parsed_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "exam_row_id": safe_str(row.get("exam_row_id", "")),
        "medical_exam_id": safe_str(row.get("medical_exam_id", "")),
        "patient_id": safe_str(row.get("patient_id", "")),
        "consultation_date": safe_str(row.get("consultation_date", "")),
        "assigned_harmful_factors": parsed_info.get("assigned_factors", []),
        "specialists": parsed_info.get("specialists", []),
        "specialist_conclusions_text": parsed_info.get("conclusions_text", ""),
    }


def predict_with_ml_stub(row: pd.Series, parsed_info: dict[str, Any]) -> dict[str, Any]:
    model_input = build_model_input(row, parsed_info)
    text = model_input["specialist_conclusions_text"].lower()
    assigned = model_input["assigned_harmful_factors"]

    if not text.strip():
        return {
            "risk_score": 0.0,
            "has_contraindications_pred": False,
            "contraindicated_factors_pred": [],
            "risk_level": "no_data",
            "model_explanation": ["ML-заглушка: отсутствует текст заключений специалистов."],
            "model_version": "stub-v0.1",
        }

    score = 0.05
    explanations = []
    predicted_factors = []

    found_terms = []
    for term in RISK_TERMS:
        if term in text:
            found_terms.append(term)
    if found_terms:
        score += min(0.55, 0.08 * len(found_terms))
        explanations.append("ML-заглушка: найдены риск-фразы: " + ", ".join(found_terms[:8]))

    for factor in assigned:
        factor_low = factor.lower()
        factor_matched = False

        for known_factor, hints in FACTOR_HINTS.items():
            if known_factor in factor_low or factor_low in known_factor:
                matched_hints = [hint for hint in hints if hint in text]
                if matched_hints:
                    factor_matched = True
                    score += 0.25
                    predicted_factors.append(factor)
                    explanations.append(
                        f"ML-заглушка: фактор '{factor}' связан с найденными признаками: "
                        + ", ".join(matched_hints)
                    )

        if not factor_matched and found_terms and re.search(r"\d", factor_low):
            score += 0.03

    target_factors = split_codes(row.get("contraindicated_factors", "")) if "contraindicated_factors" in row else []
    if target_factors:
        predicted_factors = list(dict.fromkeys(predicted_factors + target_factors))
        score = max(score, 0.85)
        explanations.append(
            "ML-заглушка: в CSV заполнена target-колонка contraindicated_factors, использую её как демо-ответ."
        )
    elif "has_contraindications" in row and safe_str(row.get("has_contraindications", "")):
        if parse_bool(row.get("has_contraindications")):
            score = max(score, 0.75)
            explanations.append("ML-заглушка: в CSV заполнена target-колонка has_contraindications=true.")

    score = max(0.0, min(0.99, score))

    if score >= 0.70:
        risk_level = "high"
    elif score >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": round(score, 3),
        "has_contraindications_pred": score >= 0.50,
        "contraindicated_factors_pred": list(dict.fromkeys(predicted_factors)),
        "risk_level": risk_level,
        "model_explanation": explanations or ["ML-заглушка: явные риск-признаки не найдены."],
        "model_version": "stub-v0.1",
    }


def predict_row_stub(row: pd.Series, *, use_ml_stub: bool = True) -> dict[str, Any]:
    parsed_info = get_parsed_info(row)
    if use_ml_stub:
        return predict_with_ml_stub(row, parsed_info)
    return {
        "risk_score": 0.0,
        "has_contraindications_pred": False,
        "contraindicated_factors_pred": [],
        "risk_level": "no_data" if not parsed_info["conclusions_text"] else "low",
        "model_explanation": ["ML-заглушка отключена. Реальная модель пока не подключена."],
        "model_version": "no-model",
    }
