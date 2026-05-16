import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir():
    sp = str(_src)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from app.profpath.dataset import make_dataset_signature, run_predictions_for_patient, sort_by_date
from app.profpath.parsing import get_row_key, safe_str, split_codes


APP_TITLE = "ПрофАрбитр AI"
STREAMLIT_REQUIRED_COLUMNS = [
    "exam_row_id",
    "patient_id",
    "consultation_date",
    "assigned_harmful_factors",
    "specialist_conclusions",
]
OPTIONAL_TARGET_COLUMNS = ["contraindicated_factors", "has_contraindications"]
STATUS_META = {
    "red": {
        "label": "Есть риск / противопоказания",
        "short": "Риск",
        "description": "Требуется ручная проверка врачом-профпатологом.",
    },
    "green": {
        "label": "Противопоказания не найдены",
        "short": "Нет риска",
        "description": "По доступным данным явные признаки противопоказаний не выделены.",
    },
    "blue": {
        "label": "Недостаточно данных",
        "short": "Нет данных",
        "description": "Не хватает заключений специалистов для уверенной оценки.",
    },
    "gray": {
        "label": "Данные загружены",
        "short": "Без ML",
        "description": "Нажмите «Предсказать», чтобы получить оценку риска.",
    },
}


st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --text: #0f172a;
        --muted: #64748b;
        --line: #dbe3ee;
        --page: #f6f8fb;
        --surface: #ffffff;
        --red: #dc2626;
        --green: #16a34a;
        --blue: #2563eb;
        --gray: #64748b;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }
    .app-title {
        font-size: 30px;
        line-height: 1.15;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 4px;
        letter-spacing: 0;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin: 0 0 18px;
    }
    .section-label {
        font-size: 15px;
        font-weight: 800;
        color: #e2e8f0;
        margin: 18px 0 8px;
    }
    .visit-card {
        border: 1px solid var(--line);
        border-left-width: 8px;
        border-radius: 8px;
        background: var(--surface);
        padding: 16px 18px;
        margin: 12px 0 8px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
    }
    .visit-card.red { border-left-color: var(--red); background: #fff7f7; }
    .visit-card.green { border-left-color: var(--green); background: #f4fbf6; }
    .visit-card.blue { border-left-color: var(--blue); background: #f5f8ff; }
    .visit-card.gray { border-left-color: var(--gray); background: #f8fafc; }
    .visit-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 14px;
        margin-bottom: 12px;
    }
    .visit-title {
        font-size: 20px;
        font-weight: 800;
        color: var(--text);
        margin: 0 0 4px;
    }
    .visit-meta {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
    }
    .status-pill {
        border-radius: 999px;
        padding: 7px 12px;
        font-weight: 800;
        font-size: 13px;
        white-space: nowrap;
    }
    .status-pill.red { color: #991b1b; background: #fee2e2; }
    .status-pill.green { color: #166534; background: #dcfce7; }
    .status-pill.blue { color: #1e40af; background: #dbeafe; }
    .status-pill.gray { color: #334155; background: #e2e8f0; }
    .card-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .info-block {
        border-top: 1px solid rgba(15, 23, 42, 0.08);
        padding-top: 10px;
        min-width: 0;
    }
    .info-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 12px;
        font-weight: 700;
        margin: 0 5px 6px 0;
        background: #eef2f7;
        color: #334155;
    }
    .badge.red { background: #fee2e2; color: #991b1b; }
    .badge.blue { background: #dbeafe; color: #1e40af; }
    .badge.green { background: #dcfce7; color: #166534; }
    .muted-text {
        color: var(--muted);
        font-size: 13px;
    }
    .timeline {
        display: flex;
        gap: 10px;
        align-items: stretch;
        overflow-x: auto;
        padding: 8px 2px 14px;
        margin-bottom: 8px;
    }
    .timeline-item {
        min-width: 145px;
        border: 1px solid var(--line);
        border-top-width: 5px;
        border-radius: 8px;
        background: #fff;
        padding: 9px 10px;
    }
    .timeline-item.red { border-top-color: var(--red); }
    .timeline-item.green { border-top-color: var(--green); }
    .timeline-item.blue { border-top-color: var(--blue); }
    .timeline-item.gray { border-top-color: var(--gray); }
    .timeline-date {
        font-size: 13px;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 3px;
    }
    .timeline-status {
        color: var(--muted);
        font-size: 12px;
    }
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0 16px;
    }
    .summary-card {
        border: 1px solid #263244;
        border-radius: 8px;
        background: #111827;
        color: #f8fafc;
        padding: 13px 14px;
        min-height: 86px;
    }
    .summary-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .summary-value {
        color: #ffffff;
        font-size: 25px;
        font-weight: 850;
        line-height: 1.1;
        overflow-wrap: anywhere;
    }
    .summary-card.red { border-top: 4px solid var(--red); }
    .summary-card.green { border-top: 4px solid var(--green); }
    .summary-card.blue { border-top: 4px solid var(--blue); }
    .summary-card.gray { border-top: 4px solid var(--gray); }
    .doctor-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 8px;
        margin: 8px 0 14px;
        align-items: start;
    }
    .doctor-card {
        border: 1px solid var(--line);
        border-left-width: 5px;
        border-radius: 8px;
        background: #ffffff;
        padding: 9px 10px;
        color: var(--text);
    }
    .doctor-card.green { border-left-color: var(--green); }
    .doctor-card.blue { border-left-color: var(--blue); }
    .doctor-card.gray { border-left-color: var(--gray); }
    .doctor-name {
        font-size: 13px;
        font-weight: 850;
        margin-bottom: 3px;
    }
    .doctor-meta {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.3;
    }
    .doctor-conclusion {
        margin-top: 6px;
        font-size: 12px;
        line-height: 1.35;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .compact-note {
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        background: #eff6ff;
        color: #1e3a8a;
        padding: 12px 14px;
        margin: 10px 0 16px;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border-color: var(--line);
    }
    @media (max-width: 900px) {
        .visit-head { display: block; }
        .status-pill { display: inline-block; margin-top: 8px; }
        .card-grid { grid-template-columns: 1fr; }
        .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(uploaded: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded)
    except UnicodeDecodeError:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="cp1251", sep=None, engine="python")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def parse_bool(value: Any) -> bool:
    text = safe_str(value).lower()
    return text in {"true", "1", "yes", "да", "истина"}


def parse_specialists(value: Any) -> list[dict[str, str]]:
    text = safe_str(value)
    if text.lower() in {"", "[]", "{}", "null", "none", "nan"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [{"specialist": "Нераспознанный текст", "conclusion": text}]
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return [{"specialist": "Нераспознанный текст", "conclusion": str(parsed)}]

    specialists: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            specialists.append({"specialist": "Запись", "conclusion": safe_str(item)})
            continue
        specialists.append(
            {
                "specialist": safe_str(item.get("specialist") or item.get("doctor") or item.get("specialty")),
                "consultation_date": safe_str(item.get("consultation_date")),
                "conclusion": safe_str(item.get("conclusion")),
                "health_group": safe_str(item.get("health_group")),
                "mkb_code": safe_str(item.get("mkb_code")),
                "mkb_description": safe_str(item.get("mkb_description")),
            }
        )
    return specialists


def specialist_dataframe(specialists: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for item in specialists:
        rows.append(
            {
                "Специалист": item.get("specialist", ""),
                "Дата": item.get("consultation_date", ""),
                "Группа": item.get("health_group", ""),
                "МКБ": item.get("mkb_code", ""),
                "Описание МКБ": item.get("mkb_description", ""),
                "Заключение": item.get("conclusion", ""),
            }
        )
    return pd.DataFrame(rows)


def doctor_marker_color(item: dict[str, str]) -> str:
    if item.get("mkb_code") and item.get("mkb_code") != "Z00.0":
        return "green"
    if item.get("health_group") or item.get("conclusion"):
        return "blue"
    return "gray"


def render_specialist_cards(specialists: list[dict[str, str]]) -> None:
    if not specialists:
        st.info("В этом профосмотре нет заключений специалистов.")
        return

    cards = []
    for item in specialists:
        color = doctor_marker_color(item)
        specialist = html.escape(item.get("specialist") or "Специалист")
        date = html.escape(format_date(item.get("consultation_date"), with_time=True))
        group = item.get("health_group")
        mkb_code = item.get("mkb_code")
        mkb_description = item.get("mkb_description")
        mkb = " ".join(x for x in [mkb_code, mkb_description] if safe_str(x))
        conclusion = item.get("conclusion") or "Без текстового заключения"
        markers = ""
        if group:
            markers += badge_html(group, "blue")
        if mkb:
            markers += badge_html(mkb, "green" if mkb_code and mkb_code != "Z00.0" else "gray")
        marker_block = f'<div style="margin-top: 6px;">{markers}</div>' if markers else ""
        cards.append(
            f'<div class="doctor-card {color}">'
            f'<div class="doctor-name">{specialist}</div>'
            f'<div class="doctor-meta">{date}</div>'
            f"{marker_block}"
            f'<div class="doctor-conclusion">{html.escape(conclusion)}</div>'
            "</div>"
        )
    st.markdown('<div class="doctor-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def first_non_empty(values: list[str]) -> str:
    for value in values:
        if safe_str(value):
            return safe_str(value)
    return ""


def diagnosis_summary(row: pd.Series, specialists: list[dict[str, str]]) -> str:
    top_level = " ".join(
        x
        for x in [
            safe_str(row.get("mkb_code", "")),
            safe_str(row.get("mkb_description", "")),
            safe_str(row.get("health_group", "")),
        ]
        if x
    )
    if top_level:
        return top_level

    specialist_diagnoses = []
    for item in specialists:
        mkb = " ".join(x for x in [item.get("mkb_code", ""), item.get("mkb_description", "")] if safe_str(x))
        if mkb and mkb not in specialist_diagnoses and mkb != "Z00.0 Общий медицинский осмотр":
            specialist_diagnoses.append(mkb)
    return "; ".join(specialist_diagnoses[:3])


def status_from_prediction(row: pd.Series, prediction: dict[str, Any] | None) -> str:
    if prediction is None:
        if safe_str(row.get("has_contraindications", "")) and parse_bool(row.get("has_contraindications")):
            return "red"
        return "gray"
    if prediction.get("risk_level") == "no_data":
        return "blue"
    if prediction.get("has_contraindications_pred"):
        return "red"
    return "green"


def clean_explanations(row: pd.Series, prediction: dict[str, Any] | None, specialists: list[dict[str, str]]) -> list[str]:
    if prediction is None:
        return ["ML-оценка еще не запускалась. Нажмите «Предсказать», чтобы получить предварительные маркеры риска."]
    if prediction.get("risk_level") == "no_data":
        return ["В этом профосмотре нет заключений специалистов или они пустые."]
    if prediction.get("has_contraindications_pred"):
        factors = split_codes(row.get("contraindicated_factors", ""))
        if prediction.get("contraindicated_factors_pred"):
            factors = list(dict.fromkeys(factors + prediction["contraindicated_factors_pred"]))
        if factors:
            return ["Найдены или предсказаны противопоказанные факторы: " + ", ".join(factors)]
        return ["Модель отметила повышенный риск. Проверьте заключения специалистов и вредные факторы."]
    non_empty = sum(1 for item in specialists if item.get("conclusion") or item.get("mkb_code"))
    return [f"Явные признаки противопоказаний не выделены. Заполненных заключений/МКБ: {non_empty}."]


def format_date(value: Any, *, with_time: bool = False) -> str:
    text = safe_str(value)
    if not text:
        return "дата не указана"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.strftime("%d.%m.%Y %H:%M" if with_time else "%d.%m.%Y")


def badge_html(text: str, color: str = "gray") -> str:
    return f'<span class="badge {color}">{html.escape(text)}</span>'


def badges(values: list[str], *, empty: str, color: str = "gray") -> str:
    cleaned = [safe_str(v) for v in values if safe_str(v)]
    if not cleaned:
        return badge_html(empty, "blue")
    return "".join(badge_html(v, color) for v in cleaned)


def render_timeline(visit_models: list[dict[str, Any]]) -> None:
    if len(visit_models) <= 1:
        return
    items = []
    for index, visit in enumerate(reversed(visit_models), start=1):
        color = visit["color"]
        items.append(
            f'<div class="timeline-item {color}">'
            f'<div class="timeline-date">Визит {index}: {html.escape(format_date(visit["row"].get("consultation_date")))}</div>'
            f'<div class="timeline-status">{html.escape(STATUS_META[color]["label"])}</div>'
            "</div>"
        )
    st.markdown('<div class="section-label">История визитов пациента</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="timeline">{"".join(items)}</div>', unsafe_allow_html=True)


def render_visit_card(visit: dict[str, Any]) -> None:
    row: pd.Series = visit["row"]
    color = visit["color"]
    specialists = visit["specialists"]
    prediction = visit["prediction"]
    assigned = split_codes(row.get("assigned_harmful_factors", ""))
    contraindicated = split_codes(row.get("contraindicated_factors", ""))
    if prediction and prediction.get("contraindicated_factors_pred"):
        contraindicated = list(dict.fromkeys(contraindicated + prediction["contraindicated_factors_pred"]))

    row_id = safe_str(row.get("exam_row_id", ""))
    exam_id = safe_str(row.get("medical_exam_id", ""))
    diagnosis = diagnosis_summary(row, specialists)
    specialist_names = [item.get("specialist", "") for item in specialists]
    score = ""
    if prediction:
        score = f"AI risk: {prediction.get('risk_score', 0):.3f} · {prediction.get('model_version', 'model')}"

    ids = [f"exam_row_id: {row_id}" if row_id else ""]
    if exam_id:
        ids.append(f"medical_exam_id: {exam_id}")
    if score:
        ids.append(score)

    st.markdown(
        f'<div class="visit-card {color}">'
        '<div class="visit-head">'
        "<div>"
        f'<div class="visit-title">Профосмотр от {html.escape(format_date(row.get("consultation_date"), with_time=True))}</div>'
        f'<div class="visit-meta">{html.escape(" · ".join(x for x in ids if x))}</div>'
        "</div>"
        f'<div class="status-pill {color}">{html.escape(STATUS_META[color]["label"])}</div>'
        "</div>"
        f'<div class="muted-text">{html.escape(STATUS_META[color]["description"])}</div>'
        '<div class="card-grid">'
        '<div class="info-block"><div class="info-label">Назначенные вредные факторы</div>'
        f'{badges(assigned, empty="не указаны")}</div>'
        '<div class="info-block"><div class="info-label">Противопоказанные факторы</div>'
        f'{badges(contraindicated, empty="не выявлены", color="red")}</div>'
        '<div class="info-block"><div class="info-label">Ключевой диагноз / МКБ</div>'
        f'{badge_html(diagnosis, "green") if diagnosis else badge_html("нет отдельного диагноза", "gray")}</div>'
        "</div>"
        '<div class="info-block" style="margin-top: 12px;"><div class="info-label">Специалисты и исследования</div>'
        f'{badges(specialist_names, empty="заключения отсутствуют")}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    render_specialist_cards(specialists)


def build_draft_conclusion(
    row: pd.Series,
    color: str,
    assigned: list[str],
    contraindicated: list[str],
    prediction: dict[str, Any] | None,
) -> str:
    patient_id = safe_str(row.get("patient_id", ""))
    date = format_date(row.get("consultation_date"), with_time=True)
    verdict = {
        "red": "Выявлены признаки возможных противопоказаний. Требуется ручная проверка врача-профпатолога.",
        "green": "По доступным данным противопоказания не выявлены. Окончательное решение принимает врач-профпатолог.",
        "blue": "Недостаточно данных для предварительной оценки. Требуется проверка заполненности заключений специалистов.",
        "gray": "ML-оценка не запускалась. Для предварительного вывода нажмите «Предсказать».",
    }[color]
    score_line = ""
    if prediction:
        score_line = f"\nAI risk score: {prediction.get('risk_score', 0):.3f}\nВерсия модели: {prediction.get('model_version', '')}"
    return (
        f"Пациент ID: {patient_id}\n"
        f"Дата профосмотра: {date}\n"
        f"exam_row_id: {safe_str(row.get('exam_row_id', ''))}\n"
        f"Назначенные вредные факторы: {', '.join(assigned) if assigned else 'не указаны'}\n"
        f"Факторы с противопоказаниями: {', '.join(contraindicated) if contraindicated else 'не выявлены'}"
        f"{score_line}\n\n"
        f"Предварительная оценка:\n{verdict}\n\n"
        "Примечание: текст является черновиком и не заменяет итоговое медицинское заключение."
    )


def build_visit_models(patient_df: pd.DataFrame, predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    visits = []
    for fallback_index, (_, row) in enumerate(patient_df.iterrows(), start=1):
        row_key = get_row_key(row, fallback_index)
        prediction = predictions.get(row_key)
        specialists = parse_specialists(row.get("specialist_conclusions", ""))
        visits.append(
            {
                "row": row,
                "row_key": row_key,
                "prediction": prediction,
                "specialists": specialists,
                "color": status_from_prediction(row, prediction),
            }
        )
    return visits


def patient_period(patient_df: pd.DataFrame) -> str:
    dates = pd.to_datetime(patient_df["consultation_date"], errors="coerce") if "consultation_date" in patient_df else pd.Series(dtype="datetime64[ns]")
    dates = dates.dropna()
    if dates.empty:
        return "даты не указаны"
    start = dates.min().strftime("%d.%m.%Y")
    end = dates.max().strftime("%d.%m.%Y")
    return start if start == end else f"{start} - {end}"


def render_summary(patient_df: pd.DataFrame, visit_models: list[dict[str, Any]], predictions: dict[str, Any]) -> None:
    counts = {color: sum(1 for visit in visit_models if visit["color"] == color) for color in STATUS_META}
    latest = visit_models[0]["color"] if visit_models else "gray"
    cards = [
        ("Профосмотров", str(len(patient_df)), "gray"),
        ("Период наблюдения", patient_period(patient_df), "gray"),
        ("Последний статус", STATUS_META[latest]["short"], latest),
        ("Риск / противопоказания", str(counts["red"]), "red"),
        ("Недостаточно данных", str(counts["blue"]), "blue"),
    ]
    html_cards = []
    for label, value, color in cards:
        html_cards.append(
            f'<div class="summary-card {color}">'
            f'<div class="summary-label">{html.escape(label)}</div>'
            f'<div class="summary-value">{html.escape(value)}</div>'
            "</div>"
        )
    st.markdown('<div class="summary-grid">' + "".join(html_cards) + "</div>", unsafe_allow_html=True)
    if not predictions:
        st.markdown(
            '<div class="compact-note">Данные пациента открыты. Нажмите «Предсказать», чтобы подсветить предварительную оценку риска.</div>',
            unsafe_allow_html=True,
        )


st.markdown(f'<div class="app-title">🩺 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">История профосмотров пациента: даты, вредные факторы, специалисты, МКБ и предварительные маркеры риска.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Данные")
    use_ml_stub = st.toggle("Использовать ML-заглушку", value=True)
    st.divider()
    st.caption("Минимальные колонки для Streamlit")
    st.code("\n".join(STREAMLIT_REQUIRED_COLUMNS), language="text")
    st.caption("Опциональные target-колонки")
    st.code("\n".join(OPTIONAL_TARGET_COLUMNS), language="text")

upload_col, _ = st.columns([2.2, 3.0], vertical_alignment="bottom")
with upload_col:
    uploaded = st.file_uploader("CSV файл", type=["csv"], label_visibility="visible")

if uploaded is None:
    st.info("Загрузите CSV и введите ID пациента. Для проверки плана можно использовать `data/train2.csv` и пациента `369710`.")
    st.stop()

try:
    df = normalize_columns(read_csv(uploaded))
except Exception as exc:
    st.error(f"Не удалось прочитать CSV: {exc}")
    st.stop()

if df.empty:
    st.warning("CSV пустой.")
    st.stop()

dataset_signature = make_dataset_signature(df)
if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state["dataset_signature"] = dataset_signature
    st.session_state["selected_patient_id"] = ""
    st.session_state["patient_predictions"] = {}
    st.session_state["predicted_patient_id"] = ""

missing = [col for col in STREAMLIT_REQUIRED_COLUMNS if col not in df.columns]
if missing:
    st.error("В CSV не хватает обязательных колонок: " + ", ".join(missing))
    st.caption("Фактические колонки: " + ", ".join(map(str, df.columns)))
    st.stop()

patient_options = sorted(df["patient_id"].dropna().astype(str).unique().tolist())
default_patient = st.session_state.get("selected_patient_id", "")
default_index = patient_options.index(default_patient) if default_patient in patient_options else None

input_col2, input_col3, input_col4 = st.columns([1.8, 0.8, 0.9], vertical_alignment="bottom")
with input_col2:
    patient_id_input = st.selectbox(
        "ID пациента",
        options=patient_options,
        index=default_index,
        placeholder="Начните вводить ID пациента",
        accept_new_options=True,
        label_visibility="visible",
    )
with input_col3:
    open_clicked = st.button("Открыть", use_container_width=True)
with input_col4:
    predict_clicked = st.button("Предсказать", type="primary", use_container_width=True)

if open_clicked:
    st.session_state["selected_patient_id"] = safe_str(patient_id_input)
    st.session_state["patient_predictions"] = {}
    st.session_state["predicted_patient_id"] = ""

selected_patient_id = safe_str(patient_id_input) or st.session_state.get("selected_patient_id", "")
if selected_patient_id:
    st.session_state["selected_patient_id"] = selected_patient_id

if not selected_patient_id:
    st.info("Введите ID пациента, чтобы открыть его профосмотры.")
    st.stop()

patient_df = df[df["patient_id"].astype(str) == selected_patient_id].copy()
if patient_df.empty:
    patient_df = df[df["patient_id"].astype(str).str.contains(selected_patient_id, case=False, na=False)].copy()

if patient_df.empty:
    st.warning(f"Пациент с ID `{selected_patient_id}` не найден в загруженном CSV.")
    st.stop()

patient_df = sort_by_date(patient_df)

if predict_clicked:
    st.session_state["selected_patient_id"] = selected_patient_id
    st.session_state["patient_predictions"] = run_predictions_for_patient(patient_df, use_ml_stub=use_ml_stub)
    st.session_state["predicted_patient_id"] = selected_patient_id
    st.rerun()

predictions: dict[str, dict[str, Any]] = {}
if st.session_state.get("predicted_patient_id") == selected_patient_id:
    predictions = st.session_state.get("patient_predictions", {})

visit_models = build_visit_models(patient_df, predictions)

st.subheader(f"Пациент ID: {selected_patient_id}")
render_summary(patient_df, visit_models, predictions)
render_timeline(visit_models)

st.markdown('<div class="section-label">Карточки профосмотров</div>', unsafe_allow_html=True)
for visit in visit_models:
    render_visit_card(visit)

st.caption(
    "Важно: это хакатонный прототип. Предварительная ML-оценка помогает подсветить риск, но итоговое решение принимает врач-профпатолог."
)
