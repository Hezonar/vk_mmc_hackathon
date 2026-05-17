from __future__ import annotations

import base64
import html
import json
from collections import Counter

from .highlighting import (
    build_highlight_links,
    card_status,
    factor_status,
    linked_factors_for_card,
)
from .recommendations import detect_research_warnings, extract_recommendations, is_research
from .types import Exam, PredictionResult, SpecialistConclusion


def esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _factor_label(status: str) -> str:
    return {
        "gray": "модель не запускалась",
        "green": "противопоказаний не найдено",
        "red": "требует внимания",
        "yellow": "ручная проверка",
        "blue": "недостаточно данных",
    }.get(status, status)


def _card_label(status: str) -> str:
    return {
        "normal": "годен",
        "warning": "проверить",
        "risk": "влияет на фактор",
        "empty": "нет данных",
    }.get(status, status)


def _kind_icon(kind: str) -> str:
    low = kind.lower()
    if "зрен" in low:
        return "👓"
    if "стомат" in low:
        return "🦷"
    if "терап" in low or "давл" in low:
        return "🫀"
    if "слух" in low:
        return "👂"
    if "экг" in low:
        return "📈"
    return "•"


def _is_general_exam_mkb(item: SpecialistConclusion) -> bool:
    return (item.mkb_code or "").strip().upper() == "Z00.0"


def _display_health_group(item: SpecialistConclusion) -> str:
    if item.health_group:
        return item.health_group
    if _is_general_exam_mkb(item):
        return "годен"
    return "—"


def _conclusion_card(idx: int, item: SpecialistConclusion, links: dict[str, list[int]], status: str) -> str:
    if not item.has_meaningful_result:
        status = "empty"
    linked = linked_factors_for_card(idx, links)
    linked_json = esc(json.dumps(linked, ensure_ascii=False))
    has_mkb = bool(item.mkb_code.strip())
    text = item.conclusion.strip() or item.mkb_description.strip() or ("Годен" if _is_general_exam_mkb(item) else "Заключение не заполнено в CSV.")
    specialist = item.specialist or "Специалист не указан"
    date = (item.consultation_date or "")[:10]
    mkb = f"{item.mkb_code} — {item.mkb_description}" if has_mkb and item.mkb_description else (item.mkb_code or "—")
    factors_badge = "".join(f'<span class="mini-factor">{esc(f)}</span>' for f in linked)
    return f"""
    <article class="med-card card-{status}" data-card-idx="{idx}" data-linked-factors='{linked_json}'>
      <div class="card-head">
        <div>
          <div class="card-title">{esc(specialist)}</div>
          <div class="card-date">{esc(date)}</div>
        </div>
        <span class="status-badge badge-{status}">{esc(_card_label(status))}</span>
      </div>
      <div class="card-meta"><b>МКБ-10:</b> {esc(mkb)}</div>
      <p class="card-text">{esc(text)}</p>
      <div class="card-footer">
        <span>Группа здоровья: <b>{esc(_display_health_group(item))}</b></span>
        <span class="linked-factors">{factors_badge}</span>
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
        title = item.specialist or "Источник не указан"
        details = []
        if item.mkb_code:
            details.append(f"МКБ-10: {item.mkb_code}")
        if item.mkb_description:
            details.append(item.mkb_description)
        text = item.conclusion or item.mkb_description or "Текст заключения не передан"
        items.append(
            {
                "title": title,
                "meta": " · ".join(details) if details else "заключение/исследование",
                "text": text[:220],
            }
        )
    if not items:
        items.append(
            {
                "title": "Связанные основания не выделены",
                "meta": "демонстрационная эвристика",
                "text": "Для этого фактора пока нет явно связанных заключений. Врач может проверить карточки вручную.",
            }
        )
    return esc(json.dumps(items, ensure_ascii=False))


def _order29n_hints_html(exam: Exam) -> str:
    factor_text = " ".join(exam.assigned_harmful_factors).upper()
    letters = []
    for letter in ["А", "К", "Ф", "Р"]:
        if letter in factor_text:
            letters.append(letter)

    letter_items = []
    if "А" in letters or "К" in letters:
        letter_items.append("для отметок А/К — врач-дерматовенеролог и врач-оториноларинголог")
    if "Р" in letters:
        letter_items.append("для отметки Р — врач-хирург")
    if "К" in letters or "Ф" in letters:
        letter_items.append("для отметок К/Ф — цифровая рентгенография легких в двух проекциях")

    if not letter_items:
        letter_items.append("в выбранных факторах не распознаны буквенные отметки А/К/Ф/Р")

    letter_html = "".join(f"<li>{esc(item)}</li>" for item in letter_items)

    return f"""
    <div class="order-grid">
      <div class="order-card">
        <div class="order-title">Возрастные уточнения</div>
        <ul>
          <li>18+ — ИМТ, ЭКГ, общий холестерин, глюкоза натощак, сердечно-сосудистый риск, флюорография/рентгенография ОГК.</li>
          <li>40+ — внутриглазное давление, абсолютный сердечно-сосудистый риск SCORE, маммография для женщин.</li>
        </ul>
      </div>
      <div class="order-card">
        <div class="order-title">Уточнения по полу</div>
        <ul>
          <li>женщинам — акушер-гинеколог, бакпосев на флору, цитология, УЗИ органов малого таза;</li>
          <li>маммография — женщинам старше 40 лет.</li>
        </ul>
      </div>
      <div class="order-card">
        <div class="order-title">Буквенные отметки у факторов</div>
        <ul>{letter_html}</ul>
      </div>
      <div class="order-card muted-order">
        <div class="order-title">Ограничение текущего CSV</div>
        <p>Возраст, пол, вид осмотра и должность не переданы отдельными колонками, поэтому блок показывает контрольные подсказки, а не автоматическое юридически значимое решение.</p>
      </div>
    </div>
    """

def _completeness_html(exam: Exam, rec_count: int, research_count: int) -> str:
    total = len(exam.specialist_conclusions)
    complete_count = sum(1 for c in exam.specialist_conclusions if c.has_meaningful_result)
    incomplete_count = total - complete_count
    mkb_count = sum(1 for c in exam.specialist_conclusions if c.mkb_code.strip())
    text_count = sum(1 for c in exam.specialist_conclusions if c.conclusion.strip())

    rows = [
        ("Факторы вредности", "есть" if exam.assigned_harmful_factors else "нет", "ok" if exam.assigned_harmful_factors else "empty"),
        ("Заключения специалистов", f"{total} записей" if total else "нет", "ok" if total else "empty"),
        ("Коды МКБ-10", f"{mkb_count} из {total}" if total else "нет", "warn" if total and mkb_count < total else ("ok" if mkb_count else "empty")),
        ("Текстовые заключения", f"{text_count} из {total}" if total else "нет", "warn" if total and text_count < total else ("ok" if text_count else "empty")),
        ("Исследования/анализы", f"{research_count} карточек" if research_count else "не выделены", "ok" if research_count else "empty"),
        ("Рекомендации", f"{rec_count} найдено" if rec_count else "не найдены", "ok" if rec_count else "empty"),
        ("ФИО / возраст / пол", "не передано в CSV", "empty"),
    ]
    if total:
        rows.insert(
            2,
            (
                "Заполненные карточки",
                f"{complete_count} из {total}",
                "ok" if incomplete_count == 0 else "empty",
            ),
        )
    html_rows = "".join(
        f"""
        <div class="check-row check-{kind}">
          <span>{esc(name)}</span>
          <b>{esc(value)}</b>
        </div>
        """
        for name, value, kind in rows
    )
    return f"<div class='check-list'>{html_rows}</div>"


def _has_enough_medical_data(exam: Exam) -> bool:
    """Demo completeness rule for the three customer scenarios.

    In the real product this should be replaced by a 29н checklist. For the MVP we only say
    that data is enough when the row has factors and at least one parsed specialist/research card.
    Missing FIO/age/sex is shown separately, but does not block the demo scenario because these
    columns are absent in the hackathon CSV.
    """
    return (
        bool(exam.assigned_harmful_factors)
        and bool(exam.specialist_conclusions)
        and all(item.has_meaningful_result for item in exam.specialist_conclusions)
    )


def _scenario_data(exam: Exam, prediction: PredictionResult) -> dict[str, str]:
    enough = _has_enough_medical_data(exam)
    if not prediction.done:
        return {
            "class": "scenario-wait",
            "number": "—",
            "title": "Сценарий еще не определен",
            "decision": "Сначала выберите профосмотр и нажмите «Предсказать».",
            "action": "До запуска модели система не формирует предварительный вывод.",
        }
    if prediction.status == "insufficient" or not enough:
        return {
            "class": "scenario-blue",
            "number": "1",
            "title": "Недостаточно медицинских данных",
            "decision": "Вынести заключение нельзя: медицинских данных недостаточно.",
            "action": "Профосмотр приостанавливается. Работнику выдается направление на дообследование. После дополнительных обследований заключение формируется повторно.",
        }
    if prediction.status == "risk":
        return {
            "class": "scenario-red",
            "number": "3",
            "title": "Выявлены противопоказания к работе",
            "decision": "Система выделила факторы, требующие проверки профпатологом и врачебной комиссией.",
            "action": "Врачебная комиссия определяет итог: годен к отдельным видам работ, постоянно непригоден или временно непригоден с возможностью повторного осмотра после лечения.",
        }
    return {
        "class": "scenario-green",
        "number": "2",
        "title": "Противопоказаний не выявлено, данных хватает",
        "decision": "По предварительному анализу противопоказания к указанным факторам не выделены.",
        "action": "Врач формирует заключение и выдает его работнику и работодателю.",
    }


def _scenario_html(exam: Exam, prediction: PredictionResult) -> str:
    data = _scenario_data(exam, prediction)
    return f"""
    <div class="scenario-box {esc(data['class'])}">
      <div class="scenario-kicker">Сценарий из презентации заказчика</div>
      <div class="scenario-head">
        <div class="scenario-number">{esc(data['number'])}</div>
        <div>
          <h3>{esc(data['title'])}</h3>
          <p>{esc(data['decision'])}</p>
        </div>
      </div>
      <div class="scenario-action">{esc(data['action'])}</div>
    </div>
    """


def _report_text_29n(exam: Exam, prediction: PredictionResult) -> str:
    date = exam.consultation_date[:10] or "не передано"
    factors = "; ".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "не передано"
    contraindicated = "; ".join(prediction.factors) if prediction.factors else "0"
    linked = prediction.linked_factors if prediction.done else {}

    evidence_lines: list[str] = []
    for factor in prediction.factors:
        indexes = linked.get(factor, [])
        if not indexes:
            evidence_lines.append(f"— фактор {factor}: связанные основания не выделены автоматически")
            continue
        for idx in indexes:
            if 0 <= idx < len(exam.specialist_conclusions):
                item = exam.specialist_conclusions[idx]
                if not item.has_meaningful_result:
                    continue
                mkb = f"; МКБ-10: {item.mkb_code}" if item.mkb_code else ""
                text = item.conclusion or item.mkb_description or "текст заключения не передан"
                evidence_lines.append(f"— фактор {factor}: {item.specialist or 'источник не указан'}{mkb}; {text}")
    if not evidence_lines:
        evidence_lines.append("— противопоказанные факторы системой не выделены")

    return (
        "ОТЧЕТ ПО РЕЗУЛЬТАТАМ ПРЕДВАРИТЕЛЬНОГО/ПЕРИОДИЧЕСКОГО МЕДИЦИНСКОГО ОСМОТРА\n"
        "Форма ориентирована на сведения заключения по Порядку 29н.\n\n"
        f"1. Дата выдачи заключения: {date}\n"
        "2. Фамилия, имя, отчество работника: не передано в CSV\n"
        "3. Дата рождения работника: не передано в CSV\n"
        "4. Пол работника: не передано в CSV\n"
        "5. Наименование работодателя: не передано в CSV\n"
        "6. Структурное подразделение работодателя: не передано в CSV\n"
        "7. Наименование должности (профессии) или вида работы: не передано в CSV\n"
        f"8. Уникальный идентификатор строки заключения: {exam.exam_row_id}\n"
        f"9. Идентификатор профосмотра: {exam.medical_exam_id or 'не передано'}\n"
        f"10. Идентификатор пациента: {exam.patient_id}\n"
        f"11. Вредные и (или) опасные производственные факторы, виды работ: {factors}\n"
        "12. Результат медицинского осмотра: медицинские противопоказания к указанным факторам/видам работ не выявлены.\n"
        f"13. Факторы, в отношении которых выявлены противопоказания: {contraindicated}\n"
        "14. Группа здоровья: указывается врачом-профпатологом/врачебной комиссией.\n\n"
        "Основания автоматизированной сводки:\n"
        + "\n".join(evidence_lines)
        + "\n\nПредседатель врачебной комиссии: __________________ / __________________\n"
        "М.П. при наличии\n\n"
        "Примечание: отчет формируется только для сценария 2 — медицинских данных хватает, противопоказания не выделены. "
        "Финальное заключение подписывает врач-профпатолог."
    )


def _report_button_html(exam: Exam, prediction: PredictionResult) -> str:
    scenario = _scenario_data(exam, prediction)
    if scenario["number"] != "2":
        return f"""
        <div class="report-dock report-disabled">
          <span>Отчет 29н доступен только для случая 2</span>
          <small>{esc(scenario['title'])}</small>
        </div>
        """
    text = _report_text_29n(exam, prediction)
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    filename = f"report_29n_{exam.exam_row_id}.txt"
    return f"""
    <div class="report-dock">
      <a class="report-button" download="{esc(filename)}" href="data:text/plain;charset=utf-8;base64,{encoded}">Скачать отчет 29н</a>
      <small>Случай 2: данных хватает, противопоказаний не выявлено</small>
    </div>
    """


def render_exam_details(exam: Exam, prediction: PredictionResult | None) -> str:
    prediction = prediction or PredictionResult()
    links = prediction.linked_factors if prediction.done else build_highlight_links(exam)
    recommendations = extract_recommendations(exam.specialist_conclusions)
    warnings = detect_research_warnings(exam.specialist_conclusions)
    complete_conclusion_count = sum(1 for c in exam.specialist_conclusions if c.has_meaningful_result)

    research_indexes = [i for i, c in enumerate(exam.specialist_conclusions) if is_research(c)]
    specialist_indexes = [i for i, c in enumerate(exam.specialist_conclusions) if i not in research_indexes]

    factor_chips = "".join(
        f"""
        <button class="factor-chip factor-{factor_status(factor, exam, prediction)}" data-factor="{esc(factor)}" data-evidence='{_factor_evidence_payload(exam, factor, links)}'>
          <span class="factor-code">{esc(factor)}</span>
          <span class="factor-state">{esc(_factor_label(factor_status(factor, exam, prediction)))}</span>
        </button>
        """
        for factor in exam.assigned_harmful_factors
    ) or "<div class='empty-card'>Факторы вредности не переданы в CSV.</div>"

    specialist_cards = "".join(
        _conclusion_card(i, exam.specialist_conclusions[i], links, card_status(i, exam.specialist_conclusions[i], prediction))
        for i in specialist_indexes
    ) or "<div class='empty-card'>Заключения специалистов не переданы в CSV.</div>"

    research_cards = "".join(
        _conclusion_card(i, exam.specialist_conclusions[i], links, card_status(i, exam.specialist_conclusions[i], prediction))
        for i in research_indexes
    ) or "<div class='empty-card'>Отдельные исследования и анализы не выделены. Система все равно ищет настораживающие признаки в текстах заключений.</div>"

    rec_cards = "".join(
        f"""
        <div class="recommendation-card">
          <div class="recommendation-kind">{esc(_kind_icon(rec.kind))} {esc(rec.kind)}</div>
          <div class="recommendation-text">{esc(rec.text)}</div>
          <div class="recommendation-source">Источник: {esc(rec.source)}</div>
        </div>
        """
        for rec in recommendations
    ) or "<div class='empty-card'>Рекомендации в текстах заключений не найдены.</div>"

    warning_cards = "".join(
        f"""
        <div class="warning-card">
          <div class="warning-kind">{esc(_kind_icon(warn.kind))} {esc(warn.kind)}</div>
          <div class="warning-text">{esc(warn.text)}</div>
          <div class="warning-source">Источник: {esc(warn.source)}</div>
        </div>
        """
        for warn in warnings[:10]
    )
    warning_section = (
        f"""
          <section class="section">
            <h2>Настораживающие признаки</h2>
            <div class="warnings-grid">{warning_cards}</div>
          </section>
        """
        if warning_cards
        else ""
    )

    csv_line = f"{exam.exam_row_id},{prediction.factors_csv if prediction.done else '—'}"
    result_class = "result-risk" if prediction.status == "risk" else "result-ok" if prediction.status == "ok" else "result-empty"
    result_title = (
        "Требуется внимание профпатолога"
        if prediction.status == "risk"
        else "Противопоказания не выделены" if prediction.status == "ok" else "Недостаточно данных" if prediction.status == "insufficient" else "Предсказание не выполнено"
    )
    factors_list = "; ".join(prediction.factors) if prediction.factors else "0"

    linked_counter = Counter()
    if prediction.done:
        for factor in prediction.factors:
            linked_counter[factor] = len(links.get(factor, []))
    linked_summary = "".join(
        f"<span class='link-summary-item'>{esc(factor)}: {count} осн.</span>"
        for factor, count in linked_counter.items()
    ) or "<span class='muted'>Связи появятся после предсказания или при наличии эвристических совпадений.</span>"

    completeness = _completeness_html(exam, len(recommendations), len(research_indexes))
    order29n_hints = _order29n_hints_html(exam)

    parse_warning_html = "".join(f"<div class='parse-warning'>{esc(w)}</div>" for w in exam.parse_warnings)

    return f"""
    <style>
      :root {{
        --bg: #f6f8fb;
        --card: #ffffff;
        --text: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --blue: #0369a1;
        --blue-soft: #e0f2fe;
        --red: #be123c;
        --red-soft: #fff1f2;
        --green: #047857;
        --green-soft: #ecfdf5;
        --yellow: #b45309;
        --yellow-soft: #fffbeb;
        --gray-soft: #f1f5f9;
      }}
      * {{ box-sizing: border-box; }}
      html {{ scroll-behavior: smooth; }}
      body {{ margin: 0; background: transparent; font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: var(--text); }}
      .exam-shell {{ min-height: 100%; overflow: visible; padding: 4px 0 18px; }}
      .exam-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 390px; gap: 16px; align-items: start; overflow: visible; }}
      .top-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 16px; align-items: stretch; }}
      .section {{ background: var(--card); border: 1px solid var(--line); border-radius: 24px; padding: 18px; box-shadow: 0 10px 24px rgba(15,23,42,.04); margin-bottom: 16px; }}
      .section-title {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom: 12px; }}
      h2 {{ font-size: 18px; line-height: 1.2; margin: 0; }}
      .hint {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
      .summary-strip {{ display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:10px; margin-bottom: 16px; }}
      .summary-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 20px; padding:14px; }}
      .summary-card span {{ display:block; color: var(--muted); font-size:12px; margin-bottom:6px; }}
      .summary-card b {{ font-size:20px; }}
      .factors {{ display:flex; flex-wrap:wrap; gap:10px; }}
      .hover-evidence {{ border:1px solid #bae6fd; background:#f0f9ff; border-radius:22px; padding:14px; min-height: 190px; }}
      .hover-evidence h3 {{ margin:0 0 8px; font-size:16px; }}
      .hover-evidence .hint {{ margin-bottom: 8px; }}
      .evidence-item {{ background:white; border:1px solid #e2e8f0; border-radius:16px; padding:10px; margin-top:8px; }}
      .evidence-item b {{ display:block; font-size:13px; color:#0f172a; }}
      .evidence-item span {{ display:block; color:#64748b; font-size:12px; margin-top:3px; }}
      .evidence-item p {{ margin:6px 0 0; font-size:12px; line-height:1.35; color:#334155; }}
      .factor-chip {{ border:1px solid var(--line); background:var(--gray-soft); border-radius:18px; min-width:112px; padding:12px; text-align:left; cursor:default; transition:.18s ease; }}
      .factor-chip:hover, .factor-chip.is-active {{ transform:translateY(-2px); box-shadow: 0 16px 28px rgba(15,23,42,.12); outline:3px solid rgba(14,165,233,.18); }}
      .factor-code {{ display:block; font-weight:900; font-size:22px; line-height:1; margin-bottom:8px; }}
      .factor-state {{ display:block; font-size:11px; line-height:1.2; font-weight:700; }}
      .factor-gray {{ background:#f8fafc; color:#475569; }}
      .factor-green {{ background:var(--green-soft); border-color:#bbf7d0; color:var(--green); }}
      .factor-red {{ background:var(--red-soft); border-color:#fecdd3; color:var(--red); }}
      .factor-yellow {{ background:var(--yellow-soft); border-color:#fde68a; color:var(--yellow); }}
      .factor-blue {{ background:var(--blue-soft); border-color:#bae6fd; color:var(--blue); }}
      .cards-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }}
      .med-card {{ border:1px solid var(--line); background:var(--card); border-radius:22px; padding:14px; transition:.18s ease; min-height: 148px; }}
      .med-card.is-highlighted {{ transform: translateY(-2px); box-shadow: 0 18px 36px rgba(15,23,42,.14); outline: 3px solid rgba(14,165,233,.24); }}
      .card-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:10px; margin-bottom:10px; }}
      .card-title {{ font-weight:800; font-size:14px; line-height:1.25; }}
      .card-date {{ color: var(--muted); font-size:12px; margin-top:3px; }}
      .status-badge {{ white-space:nowrap; border-radius:999px; padding:5px 9px; font-size:11px; font-weight:800; }}
      .badge-normal {{ background:var(--green-soft); color:var(--green); }}
      .badge-warning {{ background:var(--yellow-soft); color:var(--yellow); }}
      .badge-risk {{ background:var(--red-soft); color:var(--red); }}
      .badge-empty {{ background:var(--blue-soft); color:var(--blue); }}
      .card-normal {{ border-color:#bbf7d0; }}
      .card-warning {{ border-color:#fde68a; background:#fffdf5; }}
      .card-risk {{ border-color:#fecdd3; background:#fff7f8; }}
      .card-empty {{ border-color:#bae6fd; background:#f8fcff; }}
      .card-meta {{ color:#334155; font-size:12px; margin: 0 0 8px; }}
      .card-text {{ font-size:12.5px; line-height:1.45; margin:0; color:#334155; }}
      .card-footer {{ display:flex; justify-content:space-between; gap:8px; margin-top:12px; color:var(--muted); font-size:12px; }}
      .linked-factors {{ display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }}
      .mini-factor {{ border-radius:999px; background:#e0f2fe; color:#075985; padding:2px 7px; font-weight:800; }}
      .main-scroll {{ min-width: 0; overflow: visible; }}
      .side-sticky {{ min-width: 0; overflow: visible; }}
      .result-box {{ border-radius:24px; padding:16px; border:1px solid var(--line); }}
      .result-risk {{ background:var(--red-soft); border-color:#fecdd3; }}
      .result-ok {{ background:var(--green-soft); border-color:#bbf7d0; }}
      .result-empty {{ background:#f8fafc; }}
      .result-box h3 {{ margin:0 0 8px; font-size:18px; }}
      .result-box p {{ color:#334155; line-height:1.45; font-size:13px; margin:8px 0 0; }}
      .csv-line {{ margin-top:12px; background:#0f172a; color:#cffafe; border-radius:14px; padding:12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; overflow:auto; }}
      .link-summary {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }}
      .link-summary-item {{ background:white; border:1px solid var(--line); border-radius:999px; padding:4px 8px; font-size:12px; font-weight:800; }}
      .recommendations-grid, .warnings-grid {{ display:grid; gap:10px; }}
      .recommendation-card, .warning-card {{ border:1px solid #e2e8f0; background:#fff; border-radius:18px; padding:12px; }}
      .recommendation-kind, .warning-kind {{ font-size:12px; color:var(--muted); font-weight:800; text-transform:uppercase; letter-spacing:.02em; }}
      .recommendation-text, .warning-text {{ margin-top:7px; font-size:13px; line-height:1.45; color:#1e293b; }}
      .recommendation-source, .warning-source {{ margin-top:7px; color:var(--muted); font-size:12px; }}
      .warning-card {{ background:#fffdf5; border-color:#fde68a; }}
      .check-list {{ display:grid; gap:8px; }}
      .check-row {{ display:flex; justify-content:space-between; gap:10px; border-radius:14px; padding:10px 11px; font-size:13px; border:1px solid var(--line); }}
      .check-row b {{ text-align:right; }}
      .check-ok {{ background:var(--green-soft); border-color:#bbf7d0; }}
      .check-warn {{ background:var(--yellow-soft); border-color:#fde68a; }}
      .check-empty {{ background:var(--blue-soft); border-color:#bae6fd; }}
      .empty-card {{ border:1px dashed #cbd5e1; color:var(--muted); background:#f8fafc; border-radius:20px; padding:16px; font-size:13px; line-height:1.45; }}
      .muted {{ color:var(--muted); font-size:12px; }}
      .parse-warning {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; padding:10px 12px; border-radius:14px; margin-bottom:8px; font-size:13px; }}
      .draft {{ white-space:pre-wrap; font-size:12.5px; line-height:1.45; color:#334155; background:#f8fafc; border:1px solid var(--line); border-radius:18px; padding:12px; max-height: 420px; overflow:auto; }}

      .scenario-box {{ border-radius:24px; padding:15px; border:1px solid var(--line); margin-bottom:12px; }}
      .scenario-kicker {{ font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.04em; color:#64748b; margin-bottom:8px; }}
      .scenario-head {{ display:flex; gap:12px; align-items:flex-start; }}
      .scenario-number {{ width:34px; height:34px; flex:0 0 34px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:1000; background:white; border:1px solid rgba(15,23,42,.08); }}
      .scenario-box h3 {{ margin:0; font-size:16px; line-height:1.2; }}
      .scenario-box p {{ margin:5px 0 0; font-size:12.5px; line-height:1.4; color:#334155; }}
      .scenario-action {{ margin-top:10px; border-radius:16px; padding:10px 11px; background:rgba(255,255,255,.68); font-size:12.5px; line-height:1.4; color:#334155; }}
      .scenario-blue {{ background:#eff6ff; border-color:#bfdbfe; }}
      .scenario-green {{ background:#ecfdf5; border-color:#bbf7d0; }}
      .scenario-red {{ background:#fff1f2; border-color:#fecdd3; }}
      .scenario-wait {{ background:#f8fafc; border-color:#e2e8f0; }}
      .report-dock {{ margin-top:12px; padding:10px; border:1px solid #bae6fd; background:#f0f9ff; border-radius:18px; display:flex; flex-direction:column; gap:6px; align-items:stretch; box-shadow:none; }}
      .report-button {{ display:inline-flex; justify-content:center; align-items:center; min-height:34px; padding:7px 12px; border-radius:12px; background:#075985; color:white; text-decoration:none; font-weight:900; font-size:12px; }}
      .report-button:hover {{ background:#0c4a6e; }}
      .report-dock small {{ color:#475569; font-size:11.5px; line-height:1.3; text-align:center; }}
      .report-disabled {{ background:#f8fafc; border-color:#e2e8f0; color:#64748b; }}
      .report-disabled span {{ font-size:12px; font-weight:900; text-align:center; }}

      .order-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
      .order-card {{ border:1px solid #e2e8f0; border-radius:18px; padding:12px; background:#fff; font-size:12.5px; line-height:1.4; }}
      .order-card ul {{ margin:8px 0 0 18px; padding:0; }}
      .order-card li {{ margin-bottom:5px; }}
      .order-title {{ font-weight:900; color:#0f172a; }}
      .muted-order {{ background:#f8fafc; color:#475569; }}
      @media (max-width: 1100px) {{ .exam-grid, .top-grid {{ grid-template-columns:1fr; }} .cards-grid {{ grid-template-columns:1fr; }} .summary-strip {{ grid-template-columns:repeat(2,1fr); }} .order-grid {{ grid-template-columns:1fr; }} }}
    </style>

    <div class="exam-shell">
      {parse_warning_html}
      <div class="summary-strip">
        <div class="summary-card"><span>Идентификатор пациента</span><b>{esc(exam.patient_id)}</b></div>
        <div class="summary-card"><span>Уникальный ID строки заключения</span><b>{esc(exam.exam_row_id)}</b></div>
        <div class="summary-card"><span>Дата заключения профпатолога</span><b>{esc(exam.consultation_date[:10] or '—')}</b></div>
        <div class="summary-card"><span>Заключений профильных врачей</span><b>{complete_conclusion_count} / {len(exam.specialist_conclusions)}</b></div>
      </div>

      <div class="exam-grid">
        <main class="main-scroll">
          <div class="top-grid">
            <section class="section">
              <div class="section-title">
                <div>
                  <h2>Факторы вредности</h2>
                  <div class="hint">Наведи мышь на фактор — связанные заключения и исследования подсветятся автоматически.</div>
                </div>
              </div>
              <div class="factors">{factor_chips}</div>
            </section>

            <section class="hover-evidence" id="hoverEvidence">
              <h3>Связанные основания</h3>
              <div class="hint">Наведи на красный/желтый фактор. Здесь сразу появятся заключения и исследования, которые повлияли на подсветку — без прокрутки вниз.</div>
              <div id="hoverEvidenceItems"></div>
            </section>
          </div>

          <section class="section">
            <div class="section-title">
              <div>
                <h2>Заключения специалистов</h2>
                <div class="hint">Карточки постоянно видны после выбора осмотра. Красная подсветка появляется, если карточка связана с красным фактором.</div>
              </div>
            </div>
            <div class="cards-grid">{specialist_cards}</div>
          </section>

          <section class="section">
            <div class="section-title">
              <div>
                <h2>Исследования и анализы</h2>
                <div class="hint">Настораживающие признаки подсвечиваются даже без прямого влияния на красный фактор.</div>
              </div>
            </div>
            <div class="cards-grid">{research_cards}</div>
          </section>
        </main>

        <aside class="side-sticky">
          <section class="section">
            <h2>Вывод системы</h2>
            {_scenario_html(exam, prediction)}
            <div class="result-box {result_class}">
              <h3>{esc(result_title)}</h3>
              <div class="hint">Факторы: <b>{esc(factors_list if prediction.done else '—')}</b></div>
              <p>{esc(prediction.explanation)}</p>
              <div class="link-summary">{linked_summary}</div>
              <div class="csv-line">exam_row_id,factors<br>{esc(csv_line)}</div>
              <p><b>Важно:</b> демонстрационный вывод не является медицинским заключением. Финальное решение принимает врач-профпатолог.</p>
            </div>
          </section>

          <section class="section">
            <h2>Комплектность данных</h2>
            {completeness}
          </section>

          <section class="section">
            <h2>Контрольные подсказки по 29н</h2>
            {order29n_hints}
          </section>

          <section class="section">
            <h2>Рекомендации из заключений</h2>
            <div class="recommendations-grid">{rec_cards}</div>
          </section>

          {warning_section}

          <section class="section">
            <h2>Отчет по форме 29н</h2>
            <div class="hint">Кнопка скачивания появляется только для случая 2: медицинских данных хватает, противопоказания не выделены.</div>
            {_report_button_html(exam, prediction)}
          </section>
        </aside>
      </div>
    </div>

    <script>
      const root = document.currentScript.parentElement;
      const factors = root.querySelectorAll('.factor-chip');
      const cards = root.querySelectorAll('.med-card');

      const hoverEvidenceItems = root.querySelector('#hoverEvidenceItems');

      function renderEvidence(items) {{
        if (!hoverEvidenceItems) return;
        hoverEvidenceItems.innerHTML = (items || []).map(item =>
          '<div class="evidence-item">' +
          '<b>' + (item.title || '') + '</b>' +
          '<span>' + (item.meta || '') + '</span>' +
          '<p>' + (item.text || '') + '</p>' +
          '</div>'
        ).join('');
      }}

      function clearHighlights() {{
        factors.forEach(el => el.classList.remove('is-active'));
        cards.forEach(el => el.classList.remove('is-highlighted'));
        renderEvidence([]);
      }}

      function parseEvidence(el) {{
        try {{ return JSON.parse(el.getAttribute('data-evidence') || '[]'); }}
        catch(e) {{ return []; }}
      }}

      function parseLinked(el) {{
        try {{ return JSON.parse(el.getAttribute('data-linked-factors') || '[]'); }}
        catch(e) {{ return []; }}
      }}

      factors.forEach(factorEl => {{
        factorEl.addEventListener('mouseenter', () => {{
          clearHighlights();
          const factor = factorEl.getAttribute('data-factor');
          factorEl.classList.add('is-active');
          cards.forEach(card => {{
            const linked = parseLinked(card);
            if (linked.includes(factor)) {{
              card.classList.add('is-highlighted');
            }}
          }});
          renderEvidence(parseEvidence(factorEl));
        }});
        factorEl.addEventListener('mouseleave', clearHighlights);
      }});

      cards.forEach(card => {{
        card.addEventListener('mouseenter', () => {{
          clearHighlights();
          card.classList.add('is-highlighted');
          const linked = parseLinked(card);
          factors.forEach(factorEl => {{
            if (linked.includes(factorEl.getAttribute('data-factor'))) factorEl.classList.add('is-active');
          }});
        }});
        card.addEventListener('mouseleave', clearHighlights);
      }});
    </script>
    """


def _draft_text(exam: Exam, prediction: PredictionResult) -> str:
    date = exam.consultation_date[:10] or "не передано"
    factors = "; ".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "не передано"

    if not prediction.done:
        result = "предварительный вывод системы не сформирован"
        contraindicated = "—"
        health_group = "не определяется до проверки врачом"
    elif prediction.status == "risk":
        contraindicated = "; ".join(prediction.factors) if prediction.factors else "—"
        result = f"медицинские противопоказания требуют проверки по факторам/видам работ: {contraindicated}"
        health_group = "указывается врачебной комиссией/профпатологом"
    elif prediction.status == "ok":
        contraindicated = "0"
        result = "по демонстрационному анализу противопоказанные факторы не выделены"
        health_group = "указывается врачом"
    else:
        contraindicated = "—"
        result = "недостаточно данных для предварительного автоматизированного вывода"
        health_group = "не определяется из-за недостаточности данных"

    return (
        "ЧЕРНОВИК. НЕ ЯВЛЯЕТСЯ МЕДИЦИНСКИМ ЗАКЛЮЧЕНИЕМ.\n\n"
        "ЗАКЛЮЧЕНИЕ\n"
        "по результатам предварительного (периодического) медицинского осмотра\n\n"
        f"1. Дата выдачи заключения: {date}\n"
        "2. Фамилия, имя, отчество: не передано в CSV\n"
        "3. Дата рождения: не передано в CSV\n"
        "4. Пол: не передано в CSV\n"
        "5. Наименование работодателя: не передано в CSV\n"
        "6. Структурное подразделение, должность (профессия) или вид работы: не передано в CSV\n"
        f"7. Вредные и (или) опасные производственные факторы, виды работ: {factors}\n"
        f"8. Результат предварительного/периодического осмотра: {result}\n"
        f"9. Факторы/виды работ, в отношении которых система выделила противопоказания: {contraindicated}\n"
        f"10. Группа здоровья: {health_group}\n\n"
        "Основания для ручной проверки:\n"
        "— заключения профильных врачей и результаты исследований показаны в карточках осмотра;\n"
        "— при наведении на фактор система подсвечивает связанные основания;\n"
        "— рекомендации специалистов вынесены отдельным блоком.\n\n"
        "Председатель врачебной комиссии: __________________ / __________________\n"
        "М.П. при наличии\n\n"
        "Примечание: структура черновика ориентирована на обязательные сведения, которые указываются в заключении по Порядку 29н. "
        "Поля, отсутствующие в CSV, оставлены как «не передано в CSV»."
    )
