import io
import json
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.domain.patient import PatientRecord
from app.profpath.classification import classify_row, make_draft_conclusion
from app.profpath.constants import REQUIRED_COLUMNS
from app.profpath.dataset import run_predictions_for_patient, sort_by_date
from app.profpath.parsing import get_row_key, safe_str
from app.services.prediction import PredictionService

router = APIRouter(tags=["html"])


def _optional_int(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _optional_str(raw: str | None) -> str | None:
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _form_from_query(request: Request) -> dict[str, str]:
    qp = request.query_params
    return {
        "id": (qp.get("id") or "").strip(),
        "name": (qp.get("name") or "").strip(),
        "age": (qp.get("age") or "").strip(),
        "analyzer_id": (qp.get("analyzer_id") or "").strip(),
    }


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": "ПрофАрбитр AI", "form": _form_from_query(request)},
    )


@router.get("/favicon.ico", include_in_schema=False)
async def favicon_ico() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg", status_code=302)


@router.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    id: Annotated[str, Form()],
    name: Annotated[str | None, Form()] = None,
    age: Annotated[str | None, Form()] = None,
    analyzer_id: Annotated[str | None, Form()] = None,
) -> HTMLResponse:
    templates = request.app.state.templates
    service: PredictionService = request.app.state.prediction_service
    patient = PatientRecord(
        id=id,
        name=_optional_str(name),
        age=_optional_int(age),
    )
    aid = _optional_str(analyzer_id)
    result = service.predict(patient, analyzer_id=aid)
    vm = {
        "label": result.label.value,
        "confidence": f"{result.confidence:.0%}",
        "confidence_raw": result.confidence,
        "analyzer_id": result.analyzer_id,
        "analyzer_version": result.analyzer_version,
        "details": result.details,
    }
    return templates.TemplateResponse(
        request,
        "partials/result.html",
        {"result": vm},
    )


def _load_profpath_csv(raw: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw)
    try:
        return pd.read_csv(buf)
    except UnicodeDecodeError:
        buf.seek(0)
        return pd.read_csv(buf, encoding="cp1251", sep=None, engine="python")


def _first_existing(row: pd.Series, names: list[str]) -> str:
    for name in names:
        value = safe_str(row.get(name, ""))
        if value:
            return value
    return ""


def _specialist_summary(text: str, specialist_keywords: list[str]) -> str:
    if not text:
        return "нет данных"
    chunks = [x.strip() for x in text.replace("\r", "\n").split("\n") if x.strip()]
    lowered_keywords = [x.lower() for x in specialist_keywords]
    for chunk in chunks:
        low = chunk.lower()
        if any(keyword in low for keyword in lowered_keywords):
            return chunk[:220]
    return "нет данных"


@router.post("/profpath/cards", response_class=HTMLResponse)
async def profpath_cards(
    request: Request,
    patient_id: Annotated[str, Form()],
    file: UploadFile = File(...),
) -> HTMLResponse:
    templates = request.app.state.templates
    messages: list[str] = []
    pid = (patient_id or "").strip()
    if not pid:
        messages.append("Укажите ID пациента.")
    if not file.filename:
        messages.append("Выберите CSV файл.")
    if messages:
        return templates.TemplateResponse(
            request,
            "partials/errors.html",
            {"errors": [{"msg": m} for m in messages]},
            status_code=422,
        )
    raw = await file.read()
    if not raw:
        return templates.TemplateResponse(
            request,
            "partials/errors.html",
            {"errors": [{"msg": "Файл пустой."}]},
            status_code=422,
        )
    try:
        df = _load_profpath_csv(raw)
    except Exception:
        return templates.TemplateResponse(
            request,
            "partials/errors.html",
            {"errors": [{"msg": "Не удалось прочитать CSV. Проверьте кодировку и разделитель."}]},
            status_code=422,
        )
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return templates.TemplateResponse(
            request,
            "partials/errors.html",
            {"errors": [{"msg": "Не хватает колонок: " + ", ".join(missing)}]},
            status_code=422,
        )
    if "patient_id" not in df.columns:
        return templates.TemplateResponse(
            request,
            "partials/errors.html",
            {"errors": [{"msg": "В таблице нет колонки patient_id."}]},
            status_code=422,
        )
    patient_df = df[df["patient_id"].astype(str) == pid].copy()
    if patient_df.empty:
        patient_df = df[df["patient_id"].astype(str).str.contains(pid, case=False, na=False)].copy()
    patient_df = sort_by_date(patient_df)
    predictions = run_predictions_for_patient(patient_df, use_ml_stub=True)
    visits: list[dict[str, Any]] = []
    for fallback_index, (_, row) in enumerate(patient_df.iterrows(), start=1):
        row_key = get_row_key(row, fallback_index)
        pred = predictions.get(row_key)
        info = classify_row(row, prediction=pred)
        color = info["color"]
        badge_color = {"red": "red", "green": "green", "blue": "blue", "gray": "gray"}.get(color, "gray")
        ml = info.get("ml_result")
        ml_json = json.dumps(ml, ensure_ascii=False, indent=2) if ml else ""
        visits.append(
            {
                "patient_id": safe_str(row.get("patient_id", "")),
                "full_name": _first_existing(row, ["full_name", "fio", "name", "patient_name"]),
                "age": _first_existing(row, ["age", "patient_age"]),
                "date": safe_str(row.get("consultation_date", "")) or "дата не указана",
                "exam_id": safe_str(row.get("medical_exam_id", "")),
                "row_id": safe_str(row.get("exam_row_id", "")),
                "color": color,
                "badge_color": badge_color,
                "status": info["status"],
                "assigned": info["assigned_factors"],
                "assigned_text": ", ".join(info["assigned_factors"]) if info["assigned_factors"] else "не указано",
                "contra": info["contraindicated_factors"],
                "lor_text": _specialist_summary(info["conclusions_text"], ["лор", "оториноларинголог"]),
                "therapist_text": _specialist_summary(info["conclusions_text"], ["терапевт"]),
                "diagnosis_text": "; ".join(info["issues"][:2]) if info["issues"] else "без замечаний",
                "specialists": info["specialists"],
                "issues": info["issues"],
                "fragments": info["fragments"],
                "conclusions_text": info["conclusions_text"],
                "draft_text": make_draft_conclusion(row, info),
                "ml": ml,
                "ml_json": ml_json,
            }
        )
    return templates.TemplateResponse(
        request,
        "partials/profpath_visits.html",
        {"visits": visits},
    )
