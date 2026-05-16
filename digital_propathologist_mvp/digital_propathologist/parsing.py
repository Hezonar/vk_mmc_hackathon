from __future__ import annotations

import ast
import io
import json
from typing import Any

import pandas as pd

from .types import Exam, SpecialistConclusion

REQUIRED_COLUMNS = [
    "exam_row_id",
    "medical_exam_id",
    "patient_id",
    "consultation_date",
    "assigned_harmful_factors",
    "specialist_conclusions",
    "contraindicated_factors",
    "has_contraindications",
]

EXAM_ID_COLUMN = "Уникальный идентификатор строки заключения"


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Read CSV bytes robustly. Keeps identifiers as strings."""
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return pd.read_csv(
                io.BytesIO(data),
                encoding=encoding,
                dtype={"exam_row_id": "string", "patient_id": "string"},
                keep_default_na=False,
            )
        except Exception as exc:  # pragma: no cover - used for user files
            last_error = exc
    raise ValueError(f"Не удалось прочитать CSV: {last_error}")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["exam_row_id"] = df["exam_row_id"].astype(str).str.strip()
    df["medical_exam_id"] = df["medical_exam_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["consultation_date"] = df["consultation_date"].astype(str).str.strip()
    return df


def split_factors(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "0"}:
        return []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "да", "yes", "y", "истина"}:
        return True
    if text in {"false", "0", "нет", "no", "n", "ложь", ""}:
        return False
    return None


def _json_loads_fallback(value: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(value)
    except Exception:
        data = ast.literal_eval(value)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def parse_specialist_conclusions(value: Any) -> tuple[list[SpecialistConclusion], list[str]]:
    warnings: list[str] = []
    if value is None:
        return [], []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return [], []

    try:
        items = _json_loads_fallback(text)
    except Exception:
        warnings.append("Поле specialist_conclusions не удалось распарсить как JSON.")
        return [], warnings

    conclusions: list[SpecialistConclusion] = []
    for item in items:
        conclusions.append(
            SpecialistConclusion(
                specialist=str(item.get("specialist", "") or "").strip(),
                consultation_date=str(item.get("consultation_date", "") or "").strip(),
                conclusion=str(item.get("conclusion", "") or "").strip(),
                health_group=str(item.get("health_group", "") or "").strip(),
                mkb_code=str(item.get("mkb_code", "") or "").strip(),
                mkb_description=str(item.get("mkb_description", "") or "").strip(),
                raw=item,
            )
        )
    return conclusions, warnings


def exam_from_row(row: pd.Series | dict[str, Any]) -> Exam:
    get = row.get if isinstance(row, dict) else row.get
    conclusions, warnings = parse_specialist_conclusions(get("specialist_conclusions", ""))
    return Exam(
        exam_row_id=str(get("exam_row_id", "") or "").strip(),
        patient_id=str(get("patient_id", "") or "").strip(),
        medical_exam_id=str(get("medical_exam_id", "") or "").strip(),
        consultation_date=str(get("consultation_date", "") or "").strip(),
        assigned_harmful_factors=split_factors(get("assigned_harmful_factors", "")),
        contraindicated_factors=split_factors(get("contraindicated_factors", "")),
        has_contraindications=parse_bool(get("has_contraindications", "")),
        specialist_conclusions=conclusions,
        parse_warnings=warnings,
    )


def patient_exam_summary(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    sub = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    rows = []
    for _, row in sub.iterrows():
        conclusions, warnings = parse_specialist_conclusions(row.get("specialist_conclusions", ""))
        factors = split_factors(row.get("assigned_harmful_factors", ""))
        contraindicated = split_factors(row.get("contraindicated_factors", ""))
        med_exam_id = str(row.get("medical_exam_id", "") or "").strip()
        rows.append(
            {
                EXAM_ID_COLUMN: str(row.get("exam_row_id", "")),
                "Идентификатор профосмотра": med_exam_id if med_exam_id else "не передано",
                "Дата заключения профпатолога": str(row.get("consultation_date", ""))[:10] or "—",
                "Вредные факторы/виды работ по направлению": "; ".join(factors) if factors else "—",
                "Заключений профильных врачей": len(conclusions),
                "Комплектность данных": "ошибка JSON" if warnings else ("нет заключений" if len(conclusions) == 0 else "данные есть"),
                "Статус предсказания": "целевая колонка заполнена" if contraindicated else "не рассчитано",
            }
        )
    return pd.DataFrame(rows)


def patient_exam_summary(df: pd.DataFrame, patient_id: str) -> pd.DataFrame:
    sub = df[df["patient_id"].astype(str) == str(patient_id)].copy()
    rows = []
    for _, row in sub.iterrows():
        conclusions, warnings = parse_specialist_conclusions(row.get("specialist_conclusions", ""))
        complete_conclusions = [item for item in conclusions if item.has_meaningful_result]
        factors = split_factors(row.get("assigned_harmful_factors", ""))
        contraindicated = split_factors(row.get("contraindicated_factors", ""))
        med_exam_id = str(row.get("medical_exam_id", "") or "").strip()
        if warnings:
            completeness = "ошибка JSON"
        elif not conclusions:
            completeness = "нет заключений"
        elif len(complete_conclusions) == len(conclusions):
            completeness = "данные есть"
        else:
            completeness = "недостаточно данных"
        rows.append(
            {
                EXAM_ID_COLUMN: str(row.get("exam_row_id", "")),
                "Идентификатор профосмотра": med_exam_id if med_exam_id else "не передано",
                "Дата заключения профпатолога": str(row.get("consultation_date", ""))[:10] or "-",
                "Вредные факторы/виды работ по направлению": "; ".join(factors) if factors else "-",
                "Заключений профильных врачей": f"{len(complete_conclusions)} / {len(conclusions)}",
                "Комплектность данных": completeness,
                "Статус предсказания": "целевая колонка заполнена" if contraindicated else "не рассчитано",
            }
        )
    return pd.DataFrame(rows)
