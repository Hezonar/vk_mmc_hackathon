from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.profpath.constants import REQUIRED_COLUMNS
from app.profpath.dataset import build_patient_summary, demo_dataframe, run_predictions_for_patient, sort_by_date

router = APIRouter(prefix="/v1/profpath", tags=["profpath"])


class ProfpathPatientRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    rows: list[dict[str, Any]]
    use_ml_stub: bool = True


@router.post("/patient-summary")
def profpath_patient_summary(body: ProfpathPatientRequest) -> dict[str, Any]:
    df = pd.DataFrame(body.rows)
    if df.empty:
        raise HTTPException(status_code=422, detail="rows must not be empty")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"missing_columns": missing, "actual_columns": list(df.columns)},
        )
    selected = body.patient_id.strip()
    patient_df = df[df["patient_id"].astype(str) == selected].copy()
    if patient_df.empty:
        patient_df = df[df["patient_id"].astype(str).str.contains(selected, case=False, na=False)].copy()
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"patient_id not found: {selected!r}")
    patient_df = sort_by_date(patient_df)
    predictions = run_predictions_for_patient(patient_df, use_ml_stub=body.use_ml_stub)
    summary = build_patient_summary(patient_df, predictions)
    out_summary = summary.drop(columns=["row_key"], errors="ignore")
    return {
        "patient_id": selected,
        "predictions": predictions,
        "summary": out_summary.to_dict(orient="records"),
    }


@router.get("/demo-rows")
def profpath_demo_rows(
    patient_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    df = demo_dataframe()
    rows = df.to_dict(orient="records")
    if patient_id is None or not str(patient_id).strip():
        return {"rows": rows}
    pid = str(patient_id).strip()
    filtered = df[df["patient_id"].astype(str) == pid]
    if filtered.empty:
        filtered = df[df["patient_id"].astype(str).str.contains(pid, case=False, na=False)]
    return {"rows": filtered.to_dict(orient="records")}
