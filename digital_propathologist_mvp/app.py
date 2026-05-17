from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from digital_propathologist.exports import selected_exam_csv
from digital_propathologist.ml_model import candidate_model_available, model_metadata, predict_exam_with_candidate_model
from digital_propathologist.parsing import EXAM_ID_COLUMN, exam_from_row, normalize_dataframe, patient_exam_summary, read_csv_bytes
from digital_propathologist.review_state import (
    DAILY_TARGET,
    is_reviewed,
    load_review_state,
    mark_reviewed,
    reviewed_count,
)
from digital_propathologist.ui_html import render_exam_details

try:
    from streamlit_searchbox import st_searchbox
except Exception:  # pragma: no cover - optional dependency
    st_searchbox = None


st.set_page_config(
    page_title="Цифровой профпатолог",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  .stApp { background:#f6f7f9; color:#111827; }
  [data-testid="stHeader"] { background:rgba(246,247,249,.94); }
  .main .block-container { max-width:1520px; padding:1rem 1.25rem 2rem; }
  h1, h2, h3 { letter-spacing:0; }
  .app-title { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:10px; }
  .app-title h1 { margin:0; font-size:25px; line-height:1.15; }
  .app-title span { color:#64748b; font-size:13px; font-weight:700; }
  .side-panel, .work-panel, .metric-card {
    background:#fff; border:1px solid #d8dee8; border-radius:8px; padding:12px;
  }
  .metric-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-bottom:10px; }
  .metric-card span { display:block; color:#64748b; font-size:11px; font-weight:800; text-transform:uppercase; }
  .metric-card b { display:block; margin-top:4px; font-size:18px; color:#111827; overflow-wrap:anywhere; }
  .review-counter { border:2px solid #111827; border-radius:8px; padding:10px; margin:10px 0; background:#fff; }
  .review-counter b { font-size:20px; }
  .status-note { border:1px solid #bae6fd; background:#f0f9ff; border-radius:8px; padding:9px; color:#075985; font-size:13px; }
  .done-note { border:1px solid #bbf7d0; background:#f0fdf4; border-radius:8px; padding:9px; color:#166534; font-size:13px; margin-bottom:10px; }
  .section-divider { border:0; border-top:1px solid #d8dee8; margin:20px 0 16px; }
  .exams-section-title { margin:0 0 4px; font-size:18px; }
  .exams-section-hint { color:#64748b; font-size:13px; margin:0 0 10px; }
  div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:8px; }
  @media (max-width: 900px) { .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .app-title { align-items:flex-start; flex-direction:column; } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(data: bytes) -> pd.DataFrame:
    return normalize_dataframe(read_csv_bytes(data))


def set_dataframe(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["df"] = df
    st.session_state["source_name"] = source_name
    st.session_state["predictions"] = {}
    st.session_state.pop("selected_patient_id", None)
    st.session_state.pop("selected_exam_id", None)
    st.session_state.pop("patient_query", None)
    st.session_state.pop("patient_combo_search", None)


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df")


def patient_ids(df: pd.DataFrame) -> list[str]:
    return sorted(pid for pid in df["patient_id"].astype(str).dropna().unique().tolist() if pid)


def exam_ids_for_patient(df: pd.DataFrame, patient_id: str) -> list[str]:
    sub = df[df["patient_id"].astype(str) == str(patient_id)]
    return sub["exam_row_id"].astype(str).tolist()


def all_exam_rows(df: pd.DataFrame) -> list[tuple[str, str]]:
    rows = []
    for _, row in df.iterrows():
        patient_id = str(row.get("patient_id", "") or "").strip()
        exam_id = str(row.get("exam_row_id", "") or "").strip()
        if patient_id and exam_id:
            rows.append((patient_id, exam_id))
    return rows


def next_unreviewed(df: pd.DataFrame, current_patient: str | None = None, current_exam: str | None = None) -> tuple[str, str] | None:
    state = load_review_state()
    rows = all_exam_rows(df)
    if not rows:
        return None
    start = 0
    if current_patient and current_exam:
        try:
            start = rows.index((current_patient, current_exam)) + 1
        except ValueError:
            start = 0
    ordered = rows[start:] + rows[:start]
    for patient_id, exam_id in ordered:
        if not is_reviewed(state, patient_id, exam_id):
            return patient_id, exam_id
    return ordered[0]


def df_stats(df: pd.DataFrame | None) -> dict[str, str]:
    state = load_review_state()
    if df is None:
        return {
            "Пациентов": "-",
            "Профосмотров": "-",
            "Проверено сегодня": f"{reviewed_count(state)} / {DAILY_TARGET}",
            "Источник": "CSV не загружен",
        }
    return {
        "Пациентов": f"{df['patient_id'].nunique():,}".replace(",", " "),
        "Профосмотров": f"{len(df):,}".replace(",", " "),
        "Проверено сегодня": f"{reviewed_count(state)} / {state.get('daily_target', DAILY_TARGET)}",
        "Источник": st.session_state.get("source_name", "-"),
    }


def render_header() -> None:
    st.markdown(
        """
<div class="app-title">
  <div>
    <h1>Рабочее место врача-профпатолога</h1>
    <span>Поиск пациента, проверка профосмотра, ML-оценка факторов и формирование отчета</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metrics(df: pd.DataFrame | None) -> None:
    stats = df_stats(df)
    st.markdown(
        "<div class='metric-grid'>"
        + "".join(f"<div class='metric-card'><span>{k}</span><b>{v}</b></div>" for k, v in stats.items())
        + "</div>",
        unsafe_allow_html=True,
    )


def render_upload() -> None:
    uploaded = st.file_uploader("CSV с профосмотрами", type=["csv"], label_visibility="collapsed")
    st.caption(f"ML: {model_metadata()}")
    if uploaded is None:
        return
    uploaded_bytes = uploaded.getvalue()
    uploaded_signature = hashlib.sha256(uploaded_bytes).hexdigest()
    uploaded_id = f"{uploaded.name}:{uploaded_signature}"
    if st.session_state.get("uploaded_id") == uploaded_id:
        st.success(f"Файл загружен: {uploaded.name}")
        return
    try:
        df = load_uploaded_csv(uploaded_bytes)
    except Exception as exc:
        st.error(f"Ошибка чтения CSV: {exc}")
        return
    set_dataframe(df, uploaded.name)
    st.session_state["uploaded_id"] = uploaded_id
    first = next_unreviewed(df)
    if first:
        st.session_state["selected_patient_id"], st.session_state["selected_exam_id"] = first
    st.success(f"Файл загружен: {uploaded.name}")
    st.rerun()


def render_patient_search(df: pd.DataFrame) -> str | None:
    ids = patient_ids(df)
    selected = st.session_state.get("selected_patient_id")
    if st_searchbox is not None:
        def search_patient_ids(search_term: str) -> list[str]:
            term = (search_term or "").strip()
            if not term:
                return []
            starts = [pid for pid in ids if pid.startswith(term)]
            contains = [pid for pid in ids if term in pid and pid not in starts]
            return (starts + contains)[:12]

        found = st_searchbox(
            search_patient_ids,
            placeholder="ID пациента",
            label="Поиск пациента",
            key="patient_combo_search",
        )
        if found:
            selected = str(found)
    else:
        query = st.text_input("Поиск пациента", key="patient_query", placeholder="Введите ID или часть ID").strip()
        if query:
            exact = [pid for pid in ids if pid == query]
            options = exact or [pid for pid in ids if query in pid][:20]
            if options:
                selected = st.selectbox("Найденные пациенты", options, index=0)
            else:
                st.warning("Пациент не найден.")
    if selected:
        st.session_state["selected_patient_id"] = selected
    return selected


def render_queue(df: pd.DataFrame, selected_patient: str | None, selected_exam: str | None) -> None:
    state = load_review_state()
    st.markdown(
        f"<div class='review-counter'>Сделано сегодня:<br><b>{reviewed_count(state)}/{state.get('daily_target', DAILY_TARGET)}</b></div>",
        unsafe_allow_html=True,
    )
    if st.button("Перейти к следующему", use_container_width=True):
        nxt = next_unreviewed(df, selected_patient, selected_exam)
        if nxt:
            st.session_state["selected_patient_id"], st.session_state["selected_exam_id"] = nxt
            st.rerun()



def render_left_panel(df: pd.DataFrame | None) -> tuple[str | None, str | None]:
    with st.container(border=True):
        st.markdown("### Данные")
        render_metrics(df)
        render_upload()
    if df is None:
        st.markdown("<div class='status-note'>Загрузите CSV, чтобы открыть очередь профосмотров.</div>", unsafe_allow_html=True)
        return None, None

    with st.container(border=True):
        selected_patient = render_patient_search(df)
        if not selected_patient:
            selected_patient = st.session_state.get("selected_patient_id")
        selected_exam = st.session_state.get("selected_exam_id")
        if selected_patient:
            exams = exam_ids_for_patient(df, selected_patient)
            if exams:
                if selected_exam not in exams:
                    selected_exam = exams[0]
                selected_exam = st.selectbox(
                    "Профосмотр",
                    exams,
                    index=exams.index(selected_exam),
                    format_func=lambda value: f"№ {value}",
                )
                st.session_state["selected_exam_id"] = selected_exam
        render_queue(df, selected_patient, selected_exam)
    return selected_patient, selected_exam


def render_exam_table(summary: pd.DataFrame) -> None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<h3 class='exams-section-title'>Все профосмотры пациента</h3>", unsafe_allow_html=True)
    st.markdown("<p class='exams-section-hint'>Сводка по всем строкам заключений выбранного пациента.</p>", unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_workflow_actions(exam_id: str, exam, patient_id: str) -> None:
    predictions = st.session_state.setdefault("predictions", {})
    current = predictions.get(exam_id)
    state = load_review_state()
    already_done = is_reviewed(state, patient_id, exam_id)

    with st.container(border=True):
        st.markdown("#### Завершение проверки")
        if current and current.done:
            verdict = "не годен" if current.ml_verdict == "not_fit" else "годен" if current.ml_verdict == "fit" else "недостаточно данных"
            st.markdown(
                f"<div class='done-note'>ML-оценка: <b>{verdict}</b> · факторы <b>{current.factors_csv}</b></div>",
                unsafe_allow_html=True,
            )

        ml_col, csv_col = st.columns([1.2, 1])
        with ml_col:
            if st.button("Запустить ML-оценку", type="primary", use_container_width=True):
                if not candidate_model_available():
                    st.error("Артефакт модели 006_candidate_factor_binary не найден. Обучите модель перед запуском оценки.")
                    return
                with st.spinner("ML оценивает факторы и медицинские основания..."):
                    predictions[exam_id] = predict_exam_with_candidate_model(exam)
                st.rerun()
        with csv_col:
            st.download_button(
                "Скачать CSV",
                data=selected_exam_csv(exam_id, current),
                file_name=f"prediction_{exam_id}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if already_done:
            st.success("Этот профосмотр уже отмечен как проверенный сегодня.")
        elif st.button("Завершить проверку", type="primary", use_container_width=True):
            state, created = mark_reviewed(patient_id, exam_id)
            if created:
                st.success(f"Проверка завершена. Сделано сегодня: {reviewed_count(state)}/{state.get('daily_target', DAILY_TARGET)}.")
            else:
                st.info("Профосмотр уже был учтен ранее.")
            st.rerun()


def main() -> None:
    render_header()
    df = get_df()

    left, right = st.columns([0.24, 0.76], gap="medium")
    with left:
        patient_id, exam_id = render_left_panel(df)

    with right:
        if df is None:
            st.markdown("<div class='status-note'>Рабочая область появится после загрузки CSV.</div>", unsafe_allow_html=True)
            return
        if not patient_id or not exam_id:
            st.markdown("<div class='status-note'>Выберите пациента и профосмотр из очереди.</div>", unsafe_allow_html=True)
            return

        selected_rows = df[df["exam_row_id"].astype(str) == str(exam_id)]
        if selected_rows.empty:
            st.error("Выбранный exam_row_id не найден в загруженных данных.")
            return

        summary = patient_exam_summary(df, patient_id)
        exam = exam_from_row(selected_rows.iloc[0])
        st.markdown(f"### Пациент {patient_id}")

        prediction = st.session_state.setdefault("predictions", {}).get(exam.exam_row_id)
        html = render_exam_details(exam, prediction)
        details_height = 1180 + 110 * len(exam.specialist_conclusions) + 70 * len(exam.assigned_harmful_factors)
        components.html(html, height=min(details_height, 2400), scrolling=True)
        render_workflow_actions(exam.exam_row_id, exam, patient_id)
        render_exam_table(summary)


if __name__ == "__main__":
    main()
