from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from digital_propathologist.exports import selected_exam_csv
from digital_propathologist.parsing import EXAM_ID_COLUMN, exam_from_row, normalize_dataframe, patient_exam_summary, read_csv_bytes

try:
    from streamlit_searchbox import st_searchbox
except Exception:  # pragma: no cover - fallback for older environments
    st_searchbox = None
from digital_propathologist.prediction_stub import predict_exam
from digital_propathologist.ui_html import render_exam_details

BASE_DIR = Path(__file__).parent
DEMO_CSV = BASE_DIR / "data" / "demo_sample.csv"

st.set_page_config(
    page_title="Цифровой профпатолог — MVP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp { background: #f1f5f9; }
    [data-testid="stHeader"] { background: rgba(241,245,249,.9); }
    .main .block-container { padding-top: 1.2rem; max-width: 1440px; }
    .hero {
        background: linear-gradient(120deg, #082f49, #075985 55%, #0e7490);
        color: white;
        border-radius: 28px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow: 0 24px 60px rgba(15,23,42,.18);
    }
    .hero h1 { margin: 0; font-size: 32px; line-height: 1.1; }
    .hero p { color: #cffafe; margin: 8px 0 0; font-size: 15px; }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.2);
        color: #ecfeff;
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin-bottom: 18px; }
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(15,23,42,.04);
    }
    .metric-card span { display:block; color:#64748b; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; }
    .metric-card b { display:block; font-size:26px; line-height:1.1; margin-top:7px; color:#0f172a; }
    .panel {
        background: #fff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 16px;
        box-shadow: 0 10px 24px rgba(15,23,42,.04);
        margin-bottom: 14px;
    }
    .compact-panel { min-height: 238px; }
    .patient-panel {
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .patient-panel h3 {
        font-size: 16px;
        margin-bottom: 6px;
    }
    .patient-panel .patient-found {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 4px 10px;
        border-radius: 999px;
        background: #dcfce7;
        color: #166534;
        font-size: 12px;
        font-weight: 800;
    }
    .exam-table-wrap {
        overflow-x: auto;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        background: #fff;
    }
    table.exam-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    table.exam-table th, table.exam-table td {
        text-align: center !important;
        vertical-align: middle !important;
        padding: 12px 10px;
        border-bottom: 1px solid #e2e8f0;
    }
    table.exam-table th {
        background: #f8fafc;
        color: #334155;
        font-size: 12px;
        line-height: 1.25;
    }
    table.exam-table tr:last-child td { border-bottom: none; }
    .panel h3 { margin: 0 0 10px; font-size: 18px; }
    .soft-note {
        background: #e0f2fe;
        border: 1px solid #bae6fd;
        color: #075985;
        border-radius: 18px;
        padding: 12px 14px;
        font-size: 14px;
        line-height: 1.45;
    }
    .danger-note {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        color: #9f1239;
        border-radius: 18px;
        padding: 12px 14px;
        font-size: 14px;
        line-height: 1.45;
    }
    @media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_demo_csv() -> pd.DataFrame:
    return normalize_dataframe(read_csv_bytes(DEMO_CSV.read_bytes()))


@st.cache_data(show_spinner=False)
def load_uploaded_csv(data: bytes) -> pd.DataFrame:
    return normalize_dataframe(read_csv_bytes(data))


def set_dataframe(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["df"] = df
    st.session_state["source_name"] = source_name
    st.session_state.setdefault("predictions", {})


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df")


def df_stats(df: pd.DataFrame) -> dict[str, str]:
    contraindicated = df["contraindicated_factors"].astype(str).str.strip()
    has_any = contraindicated.ne("") & ~contraindicated.str.lower().isin(["nan", "none", "null", "0"])
    return {
        "Осмотров": f"{len(df):,}".replace(",", " "),
        "Пациентов": f"{df['patient_id'].nunique():,}".replace(",", " "),
        "С противопоказаниями": f"{int(has_any.sum()):,}".replace(",", " "),
        "Источник": st.session_state.get("source_name", "—"),
    }


def render_header() -> None:
    st.markdown(
        """
<div class="hero">
    <div class="hero-badge">MVP рабочего места врача-профпатолога</div>
    <h1>Цифровой профпатолог</h1>
    <p>Пациент → все профосмотры → раскрытие уникального ID строки заключения → факторы, заключения, исследования, рекомендации и объяснимый предварительный вывод.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metrics(df: pd.DataFrame | None) -> None:
    if df is None:
        stats = {"Осмотров": "—", "Пациентов": "—", "С противопоказаниями": "—", "Источник": "CSV не загружен"}
    else:
        stats = df_stats(df)
    st.markdown(
        "<div class='metric-grid'>"
        + "".join(f"<div class='metric-card'><span>{k}</span><b>{v}</b></div>" for k, v in stats.items())
        + "</div>",
        unsafe_allow_html=True,
    )


def render_upload_panel() -> None:
    st.markdown("<div class='panel compact-panel'><h3>1. Загрузка данных</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Загрузите CSV с осмотрами", type=["csv"], label_visibility="collapsed")
    btn_col, note_col = st.columns([0.72, 1.28])
    with btn_col:
        if st.button("Открыть демо", help="Быстро проверить интерфейс без загрузки файла"):
            df = load_demo_csv()
            set_dataframe(df, "demo_sample.csv")
            st.rerun()
    with note_col:
        st.caption("Обработка локально. ML-заглушка: prediction_stub.py")
    if uploaded is not None:
        try:
            df = load_uploaded_csv(uploaded.getvalue())
            set_dataframe(df, uploaded.name)
            st.success(f"Файл загружен: {uploaded.name}")
        except Exception as exc:
            st.error(f"Ошибка чтения CSV: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def exact_or_suggested_patient(df: pd.DataFrame, query: str) -> tuple[str | None, pd.DataFrame]:
    query = query.strip()
    if not query:
        return None, pd.DataFrame()
    exact = df[df["patient_id"].astype(str) == query]
    if not exact.empty:
        return query, exact
    suggestions = df[df["patient_id"].astype(str).str.contains(query, case=False, na=False)].head(20)
    return None, suggestions


def render_patient_panel(df: pd.DataFrame) -> str | None:
    st.markdown("<div class='panel patient-panel'><h3>2. Поиск пациента</h3>", unsafe_allow_html=True)
    patient_ids = sorted(df["patient_id"].astype(str).dropna().unique().tolist())

    def search_patient_ids(search_term: str) -> list[str]:
        term = (search_term or "").strip()
        if not term:
            return patient_ids[:10]
        starts = [pid for pid in patient_ids if pid.startswith(term)]
        contains = [pid for pid in patient_ids if term in pid and pid not in starts]
        return (starts + contains)[:10]

    patient_id = None
    if st_searchbox is not None:
        patient_id = st_searchbox(
            search_patient_ids,
            placeholder="Начните вводить ID пациента — появятся подсказки",
            label="ID пациента",
            key="patient_combo_search",
        )
    else:
        query = st.text_input(
            "ID пациента",
            placeholder="Например: 190402",
            help="Один пациент может иметь несколько профосмотров.",
        )
        exact, suggestions = exact_or_suggested_patient(df, query)
        if exact is not None:
            patient_id = exact
        elif query and not suggestions.empty:
            patient_id = st.selectbox("Подсказки по введенным цифрам", sorted(suggestions["patient_id"].astype(str).unique().tolist()))
        elif query:
            st.warning("Пациент не найден. Проверьте ID или загрузите другой CSV.")
        else:
            patient_id = st.selectbox("Быстрый выбор patient_id", patient_ids[:30], index=0 if patient_ids else None)

    if patient_id:
        count = int((df["patient_id"].astype(str) == str(patient_id)).sum())
        st.markdown(f"<span class='patient-found'>Пациент найден · профосмотров: {count}</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return str(patient_id) if patient_id else None


def render_exam_table(df: pd.DataFrame, patient_id: str) -> str | None:
    summary = patient_exam_summary(df, patient_id)
    if summary.empty:
        st.warning("У пациента нет профосмотров в загруженном CSV.")
        return None

    st.markdown(
        f"<div class='panel'><h3>3. Профосмотры пациента {patient_id}</h3><div class='soft-note'>Все ячейки выровнены по центру. Выберите нужный уникальный идентификатор строки заключения — ниже откроется витрина профосмотра.</div>",
        unsafe_allow_html=True,
    )

    options = summary[EXAM_ID_COLUMN].astype(str).tolist()
    selected_exam_id = st.selectbox(
        "Выберите уникальный идентификатор строки заключения",
        options,
        format_func=lambda value: f"{value} · {summary.loc[summary[EXAM_ID_COLUMN].astype(str) == str(value), 'Дата заключения профпатолога'].iloc[0]}",
        label_visibility="collapsed",
    )

    html_table = summary.to_html(index=False, classes="exam-table", escape=True)
    st.markdown(f"<div class='exam-table-wrap'>{html_table}</div></div>", unsafe_allow_html=True)
    return str(selected_exam_id) if selected_exam_id else None


def render_prediction_controls(exam_id: str, exam) -> None:
    predictions = st.session_state.setdefault("predictions", {})
    current = predictions.get(exam_id)

    st.markdown("<div class='panel'><h3>4. Действия по выбранному профосмотру</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.9, 0.9, 2.4])
    with col1:
        if st.button("Предсказать", type="primary"):
            with st.spinner("Модель анализирует заключения, факторы и исследования..."):
                time.sleep(1.2)
                predictions[exam_id] = predict_exam(exam)
            st.rerun()
    with col2:
        csv_data = selected_exam_csv(exam_id, current)
        st.download_button(
            "Скачать CSV-ответ",
            data=csv_data,
            file_name=f"prediction_{exam_id}.csv",
            mime="text/csv",
        )
    with col3:
        if current and current.done:
            st.markdown(
                f"<div class='soft-note'>Готово: <b>{current.factors_csv}</b>. Наведите курсор на красный фактор — связанные основания подсветятся.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='danger-note'>Предсказание еще не выполнено. ML-вывод не появляется автоматически до действия врача.</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    render_header()
    df = get_df()
    render_metrics(df)
    if df is None:
        render_upload_panel()
        st.markdown(
            "<div class='panel'><h3>Сценарий MVP</h3><div class='soft-note'>Загрузите CSV или откройте демо-данные. После этого появится поиск пациента, таблица профосмотров и раскрываемая витрина выбранного уникального ID строки заключения.</div></div>",
            unsafe_allow_html=True,
        )
        return

    top_left, top_right = st.columns([0.82, 1.18])
    with top_left:
        render_upload_panel()
    with top_right:
        patient_id = render_patient_panel(df)
    if not patient_id:
        return

    selected_exam_id = render_exam_table(df, patient_id)
    if not selected_exam_id:
        return

    selected_rows = df[df["exam_row_id"].astype(str) == str(selected_exam_id)]
    if selected_rows.empty:
        st.error("Выбранный exam_row_id не найден в dataframe.")
        return

    exam = exam_from_row(selected_rows.iloc[0])
    render_prediction_controls(exam.exam_row_id, exam)

    prediction = st.session_state.setdefault("predictions", {}).get(exam.exam_row_id)
    html = render_exam_details(exam, prediction)
    components.html(html, height=1020, scrolling=False)


if __name__ == "__main__":
    main()
