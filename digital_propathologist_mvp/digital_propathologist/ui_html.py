from __future__ import annotations

import base64
import html
import json

from .highlighting import build_highlight_links, card_status, factor_status, linked_factors_for_card
from .recommendations import detect_research_warnings, extract_recommendations, is_research
from .types import Exam, PredictionResult, SpecialistConclusion


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _factor_label(status: str) -> str:
    return {
        "gray": "ожидает ML",
        "green": "годен",
        "red": "не годен",
        "yellow": "проверить",
        "blue": "нет данных",
    }.get(status, status)


def _card_label(status: str) -> str:
    return {
        "normal": "норма",
        "warning": "проверить",
        "risk": "основание",
        "empty": "нет данных",
    }.get(status, status)


def _is_general_exam_mkb(item: SpecialistConclusion) -> bool:
    return (item.mkb_code or "").strip().upper() == "Z00.0"


def _display_health_group(item: SpecialistConclusion) -> str:
    if item.health_group:
        return item.health_group
    if _is_general_exam_mkb(item):
        return "годен"
    return "-"


def _conclusion_card(idx: int, item: SpecialistConclusion, links: dict[str, list[int]], status: str) -> str:
    if not item.has_meaningful_result:
        status = "empty"
    linked = linked_factors_for_card(idx, links)
    linked_json = esc(json.dumps(linked, ensure_ascii=False))
    text = item.conclusion.strip() or item.mkb_description.strip() or ("Годен" if _is_general_exam_mkb(item) else "Заключение не заполнено.")
    specialist = item.specialist or "Специалист не указан"
    date = (item.consultation_date or "")[:10]
    mkb = f"{item.mkb_code} - {item.mkb_description}" if item.mkb_code and item.mkb_description else (item.mkb_code or "-")
    factor_tags = "".join(f'<span class="mini-factor">{esc(f)}</span>' for f in linked)
    return f"""
    <article class="med-card card-{status}" data-card-idx="{idx}" data-linked-factors='{linked_json}'>
      <div class="card-head">
        <div>
          <div class="card-title">{esc(specialist)}</div>
          <div class="card-date">{esc(date or '-')}</div>
        </div>
        <span class="status-badge badge-{status}">{esc(_card_label(status))}</span>
      </div>
      <div class="card-meta"><b>МКБ-10:</b> {esc(mkb)}</div>
      <p class="card-text">{esc(text)}</p>
      <div class="card-footer">
        <span>Группа здоровья: <b>{esc(_display_health_group(item))}</b></span>
        <span class="linked-factors">{factor_tags}</span>
      </div>
    </article>
    """


def _factor_evidence_payload(exam: Exam, factor: str, links: dict[str, list[int]]) -> str:
    items = []
    for idx in links.get(factor, []):
        if idx < 0 or idx >= len(exam.specialist_conclusions):
            continue
        item = exam.specialist_conclusions[idx]
        if not item.has_meaningful_result:
            continue
        meta = []
        if item.mkb_code:
            meta.append(f"МКБ-10: {item.mkb_code}")
        if item.mkb_description:
            meta.append(item.mkb_description)
        items.append(
            {
                "title": item.specialist or "Источник не указан",
                "meta": " | ".join(meta) if meta else "заключение/исследование",
                "text": (item.conclusion or item.mkb_description or "Текст заключения не передан")[:240],
            }
        )
    if not items:
        items.append(
            {
                "title": "Связанные основания не выделены",
                "meta": "автоматическая проверка",
                "text": "Для этого фактора не найдено прямой связи с карточками. Проверьте заключения вручную.",
            }
        )
    return esc(json.dumps(items, ensure_ascii=False))


def _ml_title(prediction: PredictionResult) -> tuple[str, str]:
    if prediction.ml_verdict == "not_fit":
        return "Не годен по оценке ML", "result-risk"
    if prediction.ml_verdict == "fit":
        return "Годен по оценке ML", "result-ok"
    if prediction.ml_verdict == "insufficient":
        return "Недостаточно данных", "result-empty"
    return "ML-оценка не выполнена", "result-empty"


def _score_rows(prediction: PredictionResult) -> str:
    if not prediction.factor_scores:
        return "<div class='empty-card'>Скоринги факторов появятся после запуска ML.</div>"
    return "<div class='score-list'>" + "".join(
        f"""
        <div class="score-row score-{esc(str(item.get('status', 'fit')))}">
          <span>{esc(item.get('factor', '-'))}</span>
          <b>{float(item.get('score', 0)):.3f}</b>
        </div>
        """
        for item in prediction.factor_scores
    ) + "</div>"


def _report_text_29n(exam: Exam, prediction: PredictionResult) -> str:
    factors = "; ".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "не передано"
    contraindicated = "; ".join(prediction.factors) if prediction.factors else "0"
    verdict = {
        "fit": "годен по оценке ML",
        "not_fit": "не годен по оценке ML",
        "insufficient": "недостаточно данных для ML-оценки",
        "not_run": "ML-оценка не выполнена",
    }.get(prediction.ml_verdict, "ML-оценка не выполнена")
    return (
        "ОТЧЕТ ПО РЕЗУЛЬТАТАМ ПРЕДВАРИТЕЛЬНОГО/ПЕРИОДИЧЕСКОГО МЕДИЦИНСКОГО ОСМОТРА\n\n"
        f"Дата выдачи заключения: {exam.consultation_date[:10] or 'не передано'}\n"
        f"Идентификатор пациента: {exam.patient_id}\n"
        f"Идентификатор строки заключения: {exam.exam_row_id}\n"
        f"Идентификатор профосмотра: {exam.medical_exam_id or 'не передано'}\n"
        f"Вредные факторы / виды работ: {factors}\n"
        f"Оценка ML: {verdict}\n"
        f"Факторы, по которым ML выделила противопоказания: {contraindicated}\n"
        f"Версия модели: {prediction.model_version or 'не передано'}\n\n"
        "Итоговое медицинское заключение заполняет и подписывает врач-профпатолог.\n"
        "Председатель врачебной комиссии: __________________ / __________________\n"
    )


def _report_button_html(exam: Exam, prediction: PredictionResult) -> str:
    if not prediction.done:
        return "<div class='report-dock report-disabled'><span>Отчет доступен после ML-оценки</span></div>"
    encoded = base64.b64encode(_report_text_29n(exam, prediction).encode("utf-8")).decode("ascii")
    return f"""
    <div class="report-dock">
      <a class="report-button" download="report_29n_{esc(exam.exam_row_id)}.txt" href="data:text/plain;charset=utf-8;base64,{encoded}">Скачать отчет</a>
      <small>Файл содержит оценку ML и поля для итогового решения врача.</small>
    </div>
    """


def _cards_html(exam: Exam, indexes: list[int], links: dict[str, list[int]], prediction: PredictionResult, empty_text: str) -> str:
    cards = "".join(
        _conclusion_card(i, exam.specialist_conclusions[i], links, card_status(i, exam.specialist_conclusions[i], prediction))
        for i in indexes
    )
    return cards or f"<div class='empty-card'>{esc(empty_text)}</div>"


def render_exam_details(exam: Exam, prediction: PredictionResult | None) -> str:
    prediction = prediction or PredictionResult()
    links = prediction.linked_factors if prediction.done else build_highlight_links(exam)
    recommendations = extract_recommendations(exam.specialist_conclusions)
    warnings = detect_research_warnings(exam.specialist_conclusions)
    research_indexes = [i for i, c in enumerate(exam.specialist_conclusions) if is_research(c)]
    specialist_indexes = [i for i, c in enumerate(exam.specialist_conclusions) if i not in research_indexes]
    title, result_class = _ml_title(prediction)

    factor_chips = "".join(
        f"""
        <button class="factor-chip factor-{factor_status(factor, exam, prediction)}" data-factor="{esc(factor)}" data-evidence='{_factor_evidence_payload(exam, factor, links)}'>
          <span class="factor-code">{esc(factor)}</span>
          <span class="factor-state">{esc(_factor_label(factor_status(factor, exam, prediction)))}</span>
        </button>
        """
        for factor in exam.assigned_harmful_factors
    ) or "<div class='empty-card'>Факторы вредности не переданы.</div>"
    specialist_cards = _cards_html(exam, specialist_indexes, links, prediction, "Заключения специалистов не переданы.")
    research_cards = _cards_html(exam, research_indexes, links, prediction, "Исследования и анализы не выделены.")
    rec_cards = "".join(
        f"<div class='recommendation-card'><b>{esc(r.kind)}</b><p>{esc(r.text)}</p><span>{esc(r.source)}</span></div>"
        for r in recommendations
    ) or "<div class='empty-card'>Рекомендации в заключениях не найдены.</div>"
    warning_section = ""
    if warnings:
        warning_cards = "".join(
            f"<div class='warning-card'><b>{esc(w.kind)}</b><p>{esc(w.text)}</p><span>{esc(w.source)}</span></div>"
            for w in warnings
        )
        warning_section = f"<section class='section'><h2>Настораживающие признаки</h2><div class='cards-grid one'>{warning_cards}</div></section>"

    factors_list = "; ".join(prediction.factors) if prediction.factors else "0"
    csv_line = f"{exam.exam_row_id},{prediction.factors_csv if prediction.done else '-'}"
    threshold = "-" if prediction.threshold is None else f"{prediction.threshold:.2f}"
    model = prediction.model_version or "-"
    parse_warnings = "".join(f"<div class='parse-warning'>{esc(w)}</div>" for w in exam.parse_warnings)

    return f"""
    <style>
      :root {{ --ink:#111827; --muted:#64748b; --line:#d8dee8; --page:#f7f8fb; --card:#fff; --green:#15803d; --red:#dc2626; --yellow:#b45309; --blue:#0369a1; }}
      .exam-shell {{ color:var(--ink); font-family:Inter, "Segoe UI", system-ui, sans-serif; background:var(--page); padding:2px 0 8px; }}
      .summary-strip {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-bottom:10px; }}
      .summary-card, .section, .hover-evidence, .ml-footer {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
      .summary-card span, .hint, .card-date, .card-meta, .card-footer, .report-dock small, .recommendation-card span, .warning-card span {{ color:var(--muted); font-size:12px; }}
      .summary-card b {{ display:block; margin-top:4px; font-size:18px; overflow-wrap:anywhere; }}
      .work-grid {{ display:grid; grid-template-columns:minmax(0,.92fr) minmax(360px,1.08fr); gap:10px; align-items:start; }}
      .section {{ margin-bottom:10px; }}
      .section h2, .hover-evidence h3, .ml-footer h2 {{ margin:0 0 8px; font-size:16px; }}
      .top-grid {{ display:grid; grid-template-columns:1fr; gap:10px; }}
      .factors {{ display:flex; flex-wrap:wrap; gap:8px; }}
      .factor-chip {{ min-width:88px; border:2px solid var(--line); border-radius:8px; padding:10px; text-align:left; background:#f8fafc; cursor:default; transition:.16s ease; }}
      .factor-chip:hover, .factor-chip.is-active, .med-card.is-highlighted {{ transform:translateY(-1px); box-shadow:0 10px 24px rgba(15,23,42,.12); outline:2px solid rgba(2,132,199,.20); }}
      .factor-code {{ display:block; font-weight:900; font-size:21px; }}
      .factor-state {{ display:block; margin-top:5px; font-size:11px; font-weight:800; }}
      .factor-green {{ border-color:#22c55e; background:#f0fdf4; color:var(--green); }}
      .factor-red {{ border-color:#ef4444; background:#fff1f2; color:var(--red); }}
      .factor-yellow {{ border-color:#f59e0b; background:#fffbeb; color:var(--yellow); }}
      .factor-blue {{ border-color:#38bdf8; background:#f0f9ff; color:var(--blue); }}
      .cards-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
      .cards-grid.one {{ grid-template-columns:1fr; }}
      .med-card, .recommendation-card, .warning-card, .empty-card {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#fff; }}
      .card-normal {{ border-color:#22c55e; }} .card-risk {{ border-color:#ef4444; background:#fff7f7; }} .card-warning {{ border-color:#f59e0b; background:#fffdf3; }} .card-empty {{ border-color:#38bdf8; background:#f8fcff; }}
      .card-head, .card-footer, .score-row {{ display:flex; justify-content:space-between; gap:8px; align-items:flex-start; }}
      .card-title {{ font-weight:850; font-size:13px; }}
      .status-badge, .mini-factor {{ border-radius:999px; padding:3px 7px; font-size:11px; font-weight:850; background:#f1f5f9; white-space:nowrap; }}
      .badge-normal {{ color:var(--green); background:#dcfce7; }} .badge-risk {{ color:var(--red); background:#fee2e2; }} .badge-warning {{ color:var(--yellow); background:#fef3c7; }} .badge-empty {{ color:var(--blue); background:#e0f2fe; }}
      .card-text, .recommendation-card p, .warning-card p {{ font-size:12.5px; line-height:1.45; margin:7px 0 0; }}
      .linked-factors {{ display:flex; gap:4px; flex-wrap:wrap; justify-content:flex-end; }}
      .hover-evidence {{ min-height:154px; background:#f0f9ff; border-color:#bae6fd; }}
      .evidence-item {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:8px; margin-top:6px; }}
      .evidence-item b {{ font-size:12px; }} .evidence-item span {{ display:block; color:var(--muted); font-size:11px; margin-top:2px; }} .evidence-item p {{ font-size:12px; margin:5px 0 0; line-height:1.35; }}
      .ml-footer {{ margin-top:10px; }}
      .result-box {{ border:2px solid var(--line); border-radius:8px; padding:12px; }}
      .result-risk {{ border-color:#ef4444; background:#fff1f2; }} .result-ok {{ border-color:#22c55e; background:#f0fdf4; }} .result-empty {{ border-color:#94a3b8; background:#f8fafc; }}
      .result-box h3 {{ margin:0 0 6px; font-size:20px; }}
      .score-list {{ display:grid; gap:6px; margin-top:8px; }}
      .score-row {{ border:1px solid var(--line); border-radius:8px; padding:7px 9px; background:#fff; font-size:13px; }}
      .score-not_fit {{ border-color:#fecaca; color:var(--red); }} .score-fit {{ border-color:#bbf7d0; color:var(--green); }}
      .csv-line {{ margin-top:8px; background:#111827; color:#e0f2fe; border-radius:8px; padding:9px; font-family:ui-monospace,Consolas,monospace; font-size:12px; overflow:auto; }}
      .report-dock {{ margin-top:8px; display:flex; flex-direction:column; gap:5px; }} .report-button {{ display:flex; min-height:34px; align-items:center; justify-content:center; border-radius:8px; background:#111827; color:white; text-decoration:none; font-weight:850; }}
      .report-disabled {{ color:var(--muted); font-size:12px; }}
      .parse-warning {{ margin-bottom:8px; border:1px solid #fed7aa; background:#fff7ed; color:#9a3412; border-radius:8px; padding:8px; font-size:12px; }}
      @media (max-width:1100px) {{ .work-grid, .summary-strip, .cards-grid {{ grid-template-columns:1fr; }} }}
    </style>

    <div class="exam-shell">
      {parse_warnings}
      <div class="summary-strip">
        <div class="summary-card"><span>Пациент</span><b>{esc(exam.patient_id)}</b></div>
        <div class="summary-card"><span>Профосмотр</span><b>{esc(exam.medical_exam_id or '-')}</b></div>
        <div class="summary-card"><span>Строка заключения</span><b>{esc(exam.exam_row_id)}</b></div>
        <div class="summary-card"><span>Дата</span><b>{esc(exam.consultation_date[:10] or '-')}</b></div>
      </div>
      <div class="work-grid">
        <main>
          <section class="section">
            <h2>Факторы вредности</h2>
            <div class="hint">Наведение на фактор подсвечивает связанные заключения и исследования.</div>
            <div class="factors">{factor_chips}</div>
          </section>
          <section class="hover-evidence" id="hoverEvidence">
            <h3>Связанные основания</h3>
            <div class="hint">Выберите фактор курсором, чтобы увидеть медицинские основания рядом с рабочей областью.</div>
            <div id="hoverEvidenceItems"></div>
          </section>
        </main>
        <aside>
          <section class="section">
            <h2>Заключения специалистов</h2>
            <div class="cards-grid">{specialist_cards}</div>
          </section>
          <section class="section">
            <h2>Исследования и анализы</h2>
            <div class="cards-grid">{research_cards}</div>
          </section>
          <section class="section">
            <h2>Рекомендации из заключений</h2>
            <div class="cards-grid one">{rec_cards}</div>
          </section>
          {warning_section}
        </aside>
      </div>
      <section class="ml-footer">
        <h2>Итоговая оценка</h2>
        <div class="result-box {result_class}">
          <h3>{esc(title)}</h3>
          <div class="hint">ML оценила факторы: <b>{esc(factors_list)}</b></div>
          <div class="hint">Модель: <b>{esc(model)}</b> | Порог: <b>{esc(threshold)}</b></div>
          <p>{esc(prediction.explanation)}</p>
          {_score_rows(prediction)}
          <div class="csv-line">exam_row_id,factors<br>{esc(csv_line)}</div>
          {_report_button_html(exam, prediction)}
        </div>
      </section>
    </div>

    <script>
      const root = document.currentScript.parentElement;
      const factors = root.querySelectorAll('.factor-chip');
      const cards = root.querySelectorAll('.med-card');
      const hoverEvidenceItems = root.querySelector('#hoverEvidenceItems');
      function renderEvidence(items) {{
        if (!hoverEvidenceItems) return;
        hoverEvidenceItems.innerHTML = (items || []).map(item =>
          '<div class="evidence-item"><b>' + (item.title || '') + '</b><span>' +
          (item.meta || '') + '</span><p>' + (item.text || '') + '</p></div>'
        ).join('');
      }}
      function clearHighlights() {{
        factors.forEach(el => el.classList.remove('is-active'));
        cards.forEach(el => el.classList.remove('is-highlighted'));
        renderEvidence([]);
      }}
      function parseJson(el, attr) {{
        try {{ return JSON.parse(el.getAttribute(attr) || '[]'); }} catch(e) {{ return []; }}
      }}
      factors.forEach(factorEl => {{
        factorEl.addEventListener('mouseenter', () => {{
          clearHighlights();
          const factor = factorEl.getAttribute('data-factor');
          factorEl.classList.add('is-active');
          cards.forEach(card => {{
            if (parseJson(card, 'data-linked-factors').includes(factor)) card.classList.add('is-highlighted');
          }});
          renderEvidence(parseJson(factorEl, 'data-evidence'));
        }});
        factorEl.addEventListener('mouseleave', clearHighlights);
      }});
      cards.forEach(card => {{
        card.addEventListener('mouseenter', () => {{
          clearHighlights();
          card.classList.add('is-highlighted');
          const linked = parseJson(card, 'data-linked-factors');
          factors.forEach(factorEl => {{
            if (linked.includes(factorEl.getAttribute('data-factor'))) factorEl.classList.add('is-active');
          }});
        }});
        card.addEventListener('mouseleave', clearHighlights);
      }});
    </script>
    """
