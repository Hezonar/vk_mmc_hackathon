import json
from typing import Any

import pandas as pd

from app.profpath.classification import classify_row
from app.profpath.ml_stub import predict_row_stub
from app.profpath.parsing import get_row_key


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if "consultation_date" not in df.columns:
        return df
    tmp = df.copy()
    tmp["_date"] = pd.to_datetime(tmp["consultation_date"], errors="coerce", dayfirst=True)
    return tmp.sort_values(["_date", "consultation_date"], ascending=[False, False]).drop(columns=["_date"])


def make_dataset_signature(df: pd.DataFrame) -> str:
    cols = "|".join(df.columns.astype(str).tolist())
    return f"rows={len(df)}|cols={cols}"


def run_predictions_for_patient(patient_df: pd.DataFrame, *, use_ml_stub: bool = True) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for i, (_, row) in enumerate(patient_df.iterrows(), start=1):
        row_key = get_row_key(row, i)
        predictions[row_key] = predict_row_stub(row, use_ml_stub=use_ml_stub)
    return predictions


def build_patient_summary(
    patient_df: pd.DataFrame,
    predictions: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for fallback_index, (_, row) in enumerate(patient_df.iterrows(), start=1):
        row_key = get_row_key(row, fallback_index)
        info = classify_row(row, prediction=predictions.get(row_key))
        ml = info.get("ml_result") or {}
        rows.append(
            {
                "row_key": row_key,
                "exam_row_id": row.get("exam_row_id", ""),
                "medical_exam_id": row.get("medical_exam_id", ""),
                "patient_id": row.get("patient_id", ""),
                "consultation_date": row.get("consultation_date", ""),
                "status": info["status"],
                "risk_score": ml.get("risk_score", ""),
                "assigned_harmful_factors": "; ".join(info["assigned_factors"]),
                "predicted_contraindicated_factors": "; ".join(info["contraindicated_factors"]),
                "specialists_found": "; ".join(info["specialists"]),
                "n_risk_fragments": len(info["fragments"]),
            }
        )
    return pd.DataFrame(rows)


def demo_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "exam_row_id": "101",
                "medical_exam_id": "A-2023-001",
                "patient_id": "123",
                "consultation_date": "16.05.2023",
                "assigned_harmful_factors": "шум; работа на высоте",
                "specialist_conclusions": json.dumps(
                    [
                        {"specialist": "терапевт", "conclusion": "Жалоб нет. Противопоказаний не выявлено."},
                        {"specialist": "оториноларинголог", "conclusion": "Слух в пределах нормы."},
                    ],
                    ensure_ascii=False,
                ),
                "contraindicated_factors": "",
                "has_contraindications": "false",
            },
            {
                "exam_row_id": "102",
                "medical_exam_id": "A-2024-001",
                "patient_id": "123",
                "consultation_date": "16.05.2024",
                "assigned_harmful_factors": "шум; работа на высоте",
                "specialist_conclusions": json.dumps(
                    [
                        {
                            "specialist": "невролог",
                            "conclusion": "Жалобы на периодические головокружения. Рекомендовано дообследование.",
                        },
                        {
                            "specialist": "терапевт",
                            "conclusion": "Итоговое заключение требует оценки профпатолога.",
                        },
                    ],
                    ensure_ascii=False,
                ),
                "contraindicated_factors": "работа на высоте",
                "has_contraindications": "true",
            },
            {
                "exam_row_id": "103",
                "medical_exam_id": "A-2026-001",
                "patient_id": "123",
                "consultation_date": "16.05.2026",
                "assigned_harmful_factors": "шум",
                "specialist_conclusions": "",
                "contraindicated_factors": "",
                "has_contraindications": "",
            },
            {
                "exam_row_id": "104",
                "medical_exam_id": "B-2026-002",
                "patient_id": "777",
                "consultation_date": "15.05.2026",
                "assigned_harmful_factors": "химический фактор",
                "specialist_conclusions": "Врач-дерматолог: хронический дерматит, рекомендовано ограничение контакта с раздражающими веществами.",
                "contraindicated_factors": "химический фактор",
                "has_contraindications": "true",
            },
        ]
    )
