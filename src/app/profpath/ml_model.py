import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.sparse import hstack

from app.profpath.parsing import split_codes


MODEL_VERSION = "exp009_all_data_candidate_binary"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model_artifacts" / "exp009_all_data.joblib"


def _clean_token(value: str, prefix: str) -> str:
    token = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", value).strip("_")
    return f"{prefix}_{token}" if token else prefix


def _factor_family(factor: str) -> str:
    return re.split(r"[.]", factor.strip())[0]


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_specialists(value: Any) -> list[dict[str, Any]]:
    text = _safe_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_base_text(row: pd.Series, params: dict[str, Any]) -> str:
    parts: list[str] = []
    repeat_assigned = int(params["features"].get("repeat_assigned", 1))

    for factor in split_codes(row.get("assigned_harmful_factors", "")):
        parts.extend([_clean_token(factor, "factor")] * repeat_assigned)

    date = _safe_text(row.get("consultation_date", ""))
    if params["features"].get("include_month", True) and len(date) >= 7:
        parts.append("month_" + date[:7])

    for item in _parse_specialists(row.get("specialist_conclusions", "")):
        specialist = _safe_text(item.get("specialist", ""))
        health_group = _safe_text(item.get("health_group", ""))
        mkb_code = _safe_text(item.get("mkb_code", ""))
        mkb_description = _safe_text(item.get("mkb_description", ""))
        conclusion = _safe_text(item.get("conclusion", ""))

        if specialist:
            parts.append(_clean_token(specialist, "specialist"))
        if health_group:
            parts.append(_clean_token(health_group, "health"))
        if mkb_code:
            parts.extend([_clean_token(mkb_code, "mkb"), "mkb_prefix_" + mkb_code[:1].lower()])
        if mkb_description:
            parts.append(mkb_description)
        if conclusion:
            parts.append(conclusion)

    return " ".join(parts).lower()


def _candidate_text(row: pd.Series, factor: str, params: dict[str, Any]) -> str:
    candidate = _clean_token(factor, "candidate_factor")
    family = _clean_token(_factor_family(factor), "candidate_family")
    candidate_repeats = int(params["features"].get("candidate_repeats", 10))
    pair_repeats = int(params["features"].get("candidate_pair_repeats", 3))

    pieces = [_build_base_text(row, params)]
    pieces.extend([candidate] * candidate_repeats)
    pieces.extend([family] * max(1, candidate_repeats // 2))
    pieces.extend(["candidate_in_assigned_" + candidate] * pair_repeats)
    return " ".join(pieces)


def _candidate_texts(row: pd.Series, params: dict[str, Any]) -> tuple[list[str], list[str]]:
    factors = split_codes(row.get("assigned_harmful_factors", ""))
    texts = [_candidate_text(row, factor, params) for factor in factors]
    return factors, texts


@lru_cache(maxsize=1)
def load_model_artifact(model_path: str | None = None) -> dict[str, Any] | None:
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        return None
    return joblib.load(path)


def predict_with_candidate_model(row: pd.Series, *, model_path: str | None = None) -> dict[str, Any]:
    artifact = load_model_artifact(model_path)
    if artifact is None:
        raise FileNotFoundError(f"ML model artifact not found: {model_path or DEFAULT_MODEL_PATH}")

    params = artifact["params"]
    factors, texts = _candidate_texts(row, params)
    if not factors:
        return {
            "risk_score": 0.0,
            "has_contraindications_pred": False,
            "contraindicated_factors_pred": [],
            "risk_level": "no_data",
            "model_explanation": ["No assigned harmful factors were provided."],
            "model_version": artifact.get("model_version", MODEL_VERSION),
        }

    vectorizer = artifact["vectorizer"]
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    x = hstack([vectorizer["word"].transform(texts), vectorizer["char"].transform(texts)], format="csr")
    scores = model.predict_proba(x)[:, 1]
    selected = [(factor, float(score)) for factor, score in zip(factors, scores) if score >= threshold]
    selected.sort(key=lambda item: item[1], reverse=True)
    predicted_factors = [factor for factor, _ in selected]
    max_score = float(max(scores)) if len(scores) else 0.0

    if max_score >= 0.75:
        risk_level = "high"
    elif max_score >= threshold:
        risk_level = "medium"
    else:
        risk_level = "low"

    score_details = ", ".join(f"{factor}={score:.3f}" for factor, score in sorted(zip(factors, scores), key=lambda x: x[1], reverse=True)[:5])
    return {
        "risk_score": round(max_score, 3),
        "has_contraindications_pred": bool(predicted_factors),
        "contraindicated_factors_pred": predicted_factors,
        "risk_level": risk_level,
        "model_explanation": [
            "Candidate-level ML model: each assigned factor was scored independently.",
            f"Decision threshold: {threshold:.3f}. Top factor scores: {score_details or 'none'}.",
        ],
        "model_version": artifact.get("model_version", MODEL_VERSION),
    }
