from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from scipy.sparse import hstack

from .highlighting import build_highlight_links
from .types import Exam, PredictionResult

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "experiments" / "006_candidate_factor_binary" / "model.pkl"


def _clean_token(value: str, prefix: str) -> str:
    token = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", value).strip("_")
    return f"{prefix}_{token}" if token else prefix


def _factor_family(factor: str) -> str:
    return re.split(r"[.]", factor.strip())[0]


def _base_text(exam: Exam, params: dict[str, Any]) -> str:
    parts: list[str] = []
    features = params["features"]

    for factor in exam.assigned_harmful_factors:
        parts.extend([_clean_token(factor, "factor")] * int(features["repeat_assigned"]))

    date = exam.consultation_date or ""
    if bool(features["include_month"]) and len(date) >= 7:
        parts.append("month_" + date[:7])

    for item in exam.specialist_conclusions:
        if item.specialist:
            parts.append(_clean_token(item.specialist, "specialist"))
        if item.health_group:
            parts.append(_clean_token(item.health_group, "health"))
        if item.mkb_code:
            parts.extend([_clean_token(item.mkb_code, "mkb"), "mkb_prefix_" + item.mkb_code[:1].lower()])
        if item.mkb_description:
            parts.append(item.mkb_description)
        if item.conclusion:
            parts.append(item.conclusion)

    return " ".join(parts).lower()


def _candidate_text(exam: Exam, factor: str, params: dict[str, Any]) -> str:
    features = params["features"]
    candidate = _clean_token(factor, "candidate_factor")
    family = _clean_token(_factor_family(factor), "candidate_family")
    assigned = [_clean_token(x, "assigned_factor") for x in exam.assigned_harmful_factors]
    pieces = [_base_text(exam, params)]
    pieces.extend([candidate] * int(features["candidate_repeats"]))
    pieces.extend([family] * max(1, int(features["candidate_repeats"]) // 2))
    for token in assigned:
        pieces.append("assigned_context_" + token)
    for _ in range(int(features["candidate_pair_repeats"])):
        pieces.append("candidate_in_assigned_" + candidate)
    return " ".join(pieces)


@lru_cache(maxsize=1)
def load_candidate_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def candidate_model_available(path: str | Path = MODEL_PATH) -> bool:
    return Path(path).is_file()


def _insufficient_prediction(exam: Exam, reason: str) -> PredictionResult:
    return PredictionResult(
        done=True,
        factors=[],
        status="insufficient",
        explanation=reason,
        used_stub=False,
        linked_factors={},
    )


def predict_exam_with_candidate_model(exam: Exam) -> PredictionResult:
    if not exam.assigned_harmful_factors:
        return _insufficient_prediction(exam, "Недостаточно данных для ML-модели: не переданы вредные факторы.")

    meaningful = [item for item in exam.specialist_conclusions if item.has_meaningful_result]
    if not meaningful:
        return _insufficient_prediction(
            exam,
            "Недостаточно данных для ML-модели: нет заполненных заключений специалистов или результатов исследований.",
        )

    artifact = load_candidate_model()
    params = artifact["params"]
    texts = [_candidate_text(exam, factor, params) for factor in exam.assigned_harmful_factors]
    x = hstack(
        [
            artifact["word_vectorizer"].transform(texts),
            artifact["char_vectorizer"].transform(texts),
        ],
        format="csr",
    )
    scores = artifact["model"].predict_proba(x)[:, 1]
    threshold = float(artifact["factor_threshold"])
    ranked = sorted(
        zip(exam.assigned_harmful_factors, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    factors = [factor for factor, score in ranked if float(score) >= threshold]
    status = "risk" if factors else "ok"
    score_preview = ", ".join(f"{factor}: {float(score):.2f}" for factor, score in ranked[:4])
    model_version = artifact.get("model_version", "006_candidate_factor_binary")
    result = PredictionResult(
        done=True,
        factors=factors,
        status=status,
        explanation=(
            f"ML-модель {model_version} оценила назначенные факторы по текстам заключений. "
            f"Порог: {threshold:.2f}. Топ скорингов: {score_preview or 'нет факторов'}."
        ),
        used_stub=False,
    )
    result.linked_factors = build_highlight_links(exam, result)
    return result


def model_metadata(path: str | Path = MODEL_PATH) -> str:
    if not candidate_model_available(path):
        return "модель 006 не обучена"
    artifact = load_candidate_model(path)
    return json.dumps(
        {
            "model": artifact.get("model_version", "006_candidate_factor_binary"),
            "threshold": artifact.get("factor_threshold"),
            "path": str(Path(path)),
        },
        ensure_ascii=False,
    )
