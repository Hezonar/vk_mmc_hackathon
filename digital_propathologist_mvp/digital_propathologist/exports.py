from __future__ import annotations

import pandas as pd

from .types import PredictionResult


def selected_exam_csv(exam_row_id: str, prediction: PredictionResult | None) -> str:
    factors = prediction.factors_csv if prediction and prediction.done else "0"
    return f"exam_row_id,factors\n{exam_row_id},{factors}\n"


def predictions_to_csv(predictions: dict[str, PredictionResult]) -> str:
    rows = []
    for exam_row_id, pred in predictions.items():
        if pred.done:
            rows.append({"exam_row_id": exam_row_id, "factors": pred.factors_csv})
    if not rows:
        return "exam_row_id,factors\n"
    return pd.DataFrame(rows).to_csv(index=False)
