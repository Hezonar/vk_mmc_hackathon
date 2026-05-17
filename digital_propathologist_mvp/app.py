from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from digital_propathologist.exports import selected_exam_csv
from digital_propathologist.ml_model import candidate_model_available, model_metadata, predict_exam_with_candidate_model
from digital_propathologist.parsing import EXAM_ID_COLUMN, exam_from_row, normalize_dataframe, patient_exam_summary, read_csv_bytes
from digital_propathologist.ui_html import render_exam_details

try:
    from streamlit_searchbox import st_searchbox
except Exception:  # pragma: no cover - optional UI dependency
    st_searchbox = None

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
        padding: 18px;
        margin-bottom: 10px;
        min-height: 238px;
    }
    .patient-panel h3 {
        font-size: 20px;
        margin-bottom: 8px;
    }
    .patient-panel-note {
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
        margin: -2px 0 12px;
    }
    .patient-hint {
        border: 1px dashed #cbd5e1;
        border-radius: 16px;
        background: #f8fafc;
        color: #475569;
        font-size: 13px;
        line-height: 1.45;
        padding: 11px 12px;
        margin-top: 8px;
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
        margin-top: 8px;
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
def load_uploaded_csv(data: bytes) -> pd.DataFrame:
    return normalize_dataframe(read_csv_bytes(data))


def set_dataframe(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["df"] = df
    st.session_state["source_name"] = source_name
    st.session_state["predictions"] = {}
    st.session_state.pop("patient_query", None)
    st.session_state.pop("patient_combo_search", None)


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
    with st.container(border=True):
        st.markdown("### 1. Загрузка данных")
        uploaded = st.file_uploader("Загрузите CSV с осмотрами", type=["csv"])
        st.caption(f"Обработка локально. {model_metadata()}")
        if uploaded is not None:
            uploaded_bytes = uploaded.getvalue()
            uploaded_signature = hashlib.sha256(uploaded_bytes).hexdigest()
            uploaded_id = f"{uploaded.name}:{uploaded_signature}"
            already_loaded = st.session_state.get("uploaded_id") == uploaded_id
            try:
                if not already_loaded:
                    df = load_uploaded_csv(uploaded_bytes)
                    set_dataframe(df, uploaded.name)
                    st.session_state["uploaded_id"] = uploaded_id
                    st.success(f"Файл загружен: {uploaded.name}")
                    st.rerun()
                else:
                    st.success(f"Файл загружен: {uploaded.name}")
            except Exception as exc:
                st.error(f"Ошибка чтения CSV: {exc}")


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
    with st.container(border=True):
        st.markdown("### 2. Выбор пациента")
        st.caption("Введите ID полностью или несколько цифр. Подсказки появляются ниже, без автоподстановки первого пациента.")
        patient_ids = sorted(df["patient_id"].astype(str).dropna().unique().tolist())

        patient_id = None
        if st_searchbox is not None:
            def search_patient_ids(search_term: str) -> list[str]:
                term = (search_term or "").strip()
                if not term:
                    return []
                starts = [pid for pid in patient_ids if pid.startswith(term)]
                contains = [pid for pid in patient_ids if term in pid and pid not in starts]
                return (starts + contains)[:12]

            patient_id = st_searchbox(
                search_patient_ids,
                placeholder="Начните вводить ID пациента",
                label="ID пациента",
                key="patient_combo_search",
            )
        else:
            query = st.text_input(
                "ID пациента",
                key="patient_query",
                placeholder="Например: 310942",
                help="Можно ввести часть ID и выбрать найденную подсказку.",
            ).strip()
            exact, suggestions = exact_or_suggested_patient(df, query)
            if exact is not None:
                patient_id = exact
            elif query and not suggestions.empty:
                options = sorted(suggestions["patient_id"].astype(str).unique().tolist())
                patient_id = st.selectbox(
                    "Найденные совпадения",
                    options,
                    index=None,
                    placeholder="Выберите patient_id",
                )
            elif query:
                st.warning("Пациент не найден. Проверьте ID или загрузите другой CSV.")

        if not patient_id:
            sample_ids = ", ".join(patient_ids[:5])
            patient_count = f"{len(patient_ids):,}".replace(",", " ")
            st.markdown(
                f"<div class='patient-hint'>В датасете {patient_count} пациентов. Примеры: {sample_ids}</div>",
                unsafe_allow_html=True,
            )

        if patient_id:
            count = int((df["patient_id"].astype(str) == str(patient_id)).sum())
            st.markdown(f"<span class='patient-found'>Пациент найден · профосмотров: {count}</span>", unsafe_allow_html=True)
    return str(patient_id) if patient_id else None


def render_exam_selector(df: pd.DataFrame, patient_id: str) -> tuple[str | None, pd.DataFrame]:
    summary = patient_exam_summary(df, patient_id)
    if summary.empty:
        st.warning("У пациента нет профосмотров в загруженном CSV.")
        return None, summary

    with st.container(border=True):
        st.markdown("### 3. Актуальное исследование")
        st.caption("Выберите уникальный идентификатор строки заключения, который нужно разобрать сейчас.")
        options = summary[EXAM_ID_COLUMN].astype(str).tolist()
        selected_exam_id = st.selectbox(
            "Уникальный идентификатор строки заключения",
            options,
            format_func=lambda value: f"{value} · {summary.loc[summary[EXAM_ID_COLUMN].astype(str) == str(value), 'Дата заключения профпатолога'].iloc[0]}",
        )
    return (str(selected_exam_id) if selected_exam_id else None), summary


def render_exam_table(summary: pd.DataFrame, patient_id: str) -> None:
    st.markdown(
        f"<div class='panel'><h3>5. Профосмотры пациента {patient_id}</h3><div class='soft-note'>Справочный список всех профосмотров пациента. Актуальное исследование выбрано выше.</div>",
        unsafe_allow_html=True,
    )
    html_table = summary.to_html(index=False, classes="exam-table", escape=True)
    st.markdown(f"<div class='exam-table-wrap'>{html_table}</div></div>", unsafe_allow_html=True)


def render_prediction_controls(exam_id: str, exam) -> None:
    predictions = st.session_state.setdefault("predictions", {})
    current = predictions.get(exam_id)

    st.markdown("<div class='panel'><h3>4. Действия по выбранному профосмотру</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.9, 0.9, 2.4])
    with col1:
        if st.button("Предсказать", type="primary"):
            if not candidate_model_available():
                st.error("Артефакт модели 006 не найден. Запустите experiments/006_candidate_factor_binary/train.py.")
                return
            with st.spinner("Модель анализирует заключения, факторы и исследования..."):
                predictions[exam_id] = predict_exam_with_candidate_model(exam)
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
                "<div class='danger-note'>Предсказание еще не выполнено. ML-модель 006 запускается только по действию врача.</div>",
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
            "<div class='panel'><h3>Сценарий MVP</h3><div class='soft-note'>Загрузите CSV. После этого появится поиск пациента, выбор актуального исследования и витрина выбранного уникального ID строки заключения.</div></div>",
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

    selected_exam_id, exam_summary = render_exam_selector(df, patient_id)
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
    details_height = 1180 + 130 * len(exam.specialist_conclusions) + 80 * len(exam.assigned_harmful_factors)
    components.html(html, height=min(details_height, 2600), scrolling=True)
    render_exam_table(exam_summary, patient_id)


if __name__ == "__main__":
    main()
