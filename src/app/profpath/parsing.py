import json
import re
from typing import Any

import pandas as pd

from app.profpath.constants import NO_DATA_MARKERS, RISK_TERMS


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_bool(x: Any) -> bool:
    s = safe_str(x).lower()
    if s in {"true", "1", "yes", "да", "истина"}:
        return True
    if s in {"false", "0", "no", "нет", "ложь", "", "nan", "none"}:
        return False
    return bool(s)


def split_codes(value: Any) -> list[str]:
    if isinstance(value, list):
        raw = value
    else:
        s = safe_str(value)
        if s.lower() in NO_DATA_MARKERS:
            return []
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                raw = obj
            elif isinstance(obj, dict):
                raw = list(obj.values())
            else:
                raw = [str(obj)]
        except Exception:
            raw = re.split(r"[;,|\n]+", s)

    result = []
    for item in raw:
        item_s = safe_str(item)
        if item_s and item_s.lower() not in NO_DATA_MARKERS:
            result.append(item_s)
    return result


def flatten_json_like(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        chunks = []
        for k, v in obj.items():
            chunks.append(f"{k}: {flatten_json_like(v)}")
        return "; ".join(chunks)
    if isinstance(obj, list):
        return "\n".join(flatten_json_like(v) for v in obj)
    return str(obj)


def parse_specialist_conclusions(value: Any) -> tuple[str, list[str]]:
    s = safe_str(value)
    if s.lower() in NO_DATA_MARKERS:
        return "", []

    specialists = []
    text = s

    try:
        obj = json.loads(s)
        text = flatten_json_like(obj)

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    for key in [
                        "specialist",
                        "doctor",
                        "doctor_type",
                        "speciality",
                        "specialty",
                        "Врач",
                        "Специалист",
                    ]:
                        if key in item and safe_str(item[key]):
                            specialists.append(safe_str(item[key]))
        elif isinstance(obj, dict):
            for key in [
                "specialist",
                "doctor",
                "doctor_type",
                "speciality",
                "specialty",
                "Врач",
                "Специалист",
            ]:
                if key in obj and safe_str(obj[key]):
                    specialists.append(safe_str(obj[key]))
    except Exception:
        pass

    common = [
        "терапевт",
        "невролог",
        "оториноларинголог",
        "лор",
        "офтальмолог",
        "психиатр",
        "дерматолог",
        "хирург",
        "кардиолог",
        "эндокринолог",
        "профпатолог",
        "гинеколог",
        "стоматолог",
    ]
    low = text.lower()
    for sp in common:
        if sp in low and sp not in [x.lower() for x in specialists]:
            specialists.append(sp)

    return text, specialists


def extract_problem_fragments(text: str, max_items: int = 8) -> list[str]:
    if not text:
        return []
    low = text.lower()
    fragments = []
    for term in RISK_TERMS:
        idx = low.find(term)
        if idx >= 0:
            start = max(0, idx - 70)
            end = min(len(text), idx + 120)
            frag = text[start:end].replace("\n", " ").strip()
            fragments.append(frag)
    seen = set()
    unique = []
    for f in fragments:
        key = f.lower()
        if key not in seen:
            unique.append(f)
            seen.add(key)
    return unique[:max_items]


def get_row_key(row: pd.Series, fallback_index: Any) -> str:
    exam_row_id = safe_str(row.get("exam_row_id", ""))
    if exam_row_id:
        return f"exam_row_id::{exam_row_id}"
    medical_exam_id = safe_str(row.get("medical_exam_id", ""))
    patient_id = safe_str(row.get("patient_id", ""))
    date = safe_str(row.get("consultation_date", ""))
    return f"row::{fallback_index}::{medical_exam_id}::{patient_id}::{date}"


def get_parsed_info(row: pd.Series) -> dict[str, Any]:
    conclusions_text, specialists = parse_specialist_conclusions(row.get("specialist_conclusions", ""))
    assigned = split_codes(row.get("assigned_harmful_factors", ""))
    fragments = extract_problem_fragments(conclusions_text)
    return {
        "assigned_factors": assigned,
        "conclusions_text": conclusions_text,
        "specialists": specialists,
        "fragments": fragments,
    }
