from typing import Any

import pandas as pd

from app.profpath.parsing import get_parsed_info, safe_str


def classify_row(row: pd.Series, prediction: dict[str, Any] | None = None) -> dict[str, Any]:
    parsed_info = get_parsed_info(row)
    assigned = parsed_info["assigned_factors"]
    conclusions_text = parsed_info["conclusions_text"]
    specialists = parsed_info["specialists"]
    fragments = parsed_info["fragments"]

    if prediction is None:
        return {
            "status": "Данные загружены",
            "color": "gray",
            "issues": [
                "Данные по посещению загружены. Для ML-оценки нажмите кнопку «Предсказать» рядом с ID пациента."
            ],
            "assigned_factors": assigned,
            "contraindicated_factors": [],
            "conclusions_text": conclusions_text,
            "specialists": specialists,
            "fragments": fragments,
            "ml_result": None,
        }

    issues = []
    if prediction["risk_level"] == "no_data":
        status = "Нет данных"
        color = "blue"
        issues.append("Не найдены заключения врачей-специалистов по этому профосмотру.")
        issues.extend(prediction.get("model_explanation", []))
    elif prediction["has_contraindications_pred"]:
        status = "Есть риски / противопоказания"
        color = "red"
        issues.extend(prediction.get("model_explanation", []))
        if prediction.get("contraindicated_factors_pred"):
            issues.append(
                "Предсказанные противопоказанные факторы: "
                + ", ".join(prediction["contraindicated_factors_pred"])
            )
    else:
        status = "Нет противопоказаний"
        color = "green"
        issues.extend(prediction.get("model_explanation", []))

    return {
        "status": status,
        "color": color,
        "issues": issues,
        "assigned_factors": assigned,
        "contraindicated_factors": prediction.get("contraindicated_factors_pred", []),
        "conclusions_text": conclusions_text,
        "specialists": specialists,
        "fragments": fragments,
        "ml_result": prediction,
    }


def make_draft_conclusion(row: pd.Series, info: dict[str, Any]) -> str:
    patient_id = safe_str(row.get("patient_id", "не указан"))
    date = safe_str(row.get("consultation_date", "не указана"))
    assigned = ", ".join(info["assigned_factors"]) if info["assigned_factors"] else "не указаны"
    contra = ", ".join(info["contraindicated_factors"]) if info["contraindicated_factors"] else "не выявлены"
    issues = "\n".join(f"- {x}" for x in info["issues"])

    ml_part = ""
    if info.get("ml_result"):
        ml = info["ml_result"]
        ml_part = f"\nAI risk score: {ml['risk_score']}\nВерсия модели: {ml['model_version']}\n"

    if info["color"] == "gray":
        verdict = "ML-предсказание ещё не выполнено. Нажмите кнопку «Предсказать»."
    elif info["color"] == "red":
        verdict = "Выявлены риски/признаки возможных противопоказаний. Требуется ручная проверка врача-профпатолога."
    elif info["color"] == "blue":
        verdict = "Недостаточно данных для предварительной оценки. Требуется загрузка/проверка заключений специалистов."
    else:
        verdict = "По представленным данным противопоказания не выявлены. Окончательное решение принимает врач-профпатолог."

    return (
        f"Пациент ID: {patient_id}\n"
        f"Дата профосмотра: {date}\n"
        f"Назначенные вредные факторы / виды работ: {assigned}\n"
        f"Факторы с выявленными/предсказанными противопоказаниями: {contra}\n"
        f"{ml_part}\n"
        f"Автоматизированная предварительная оценка:\n{verdict}\n\n"
        f"Основания/замечания:\n{issues}\n\n"
        f"Примечание: данный текст является черновиком для врача-профпатолога и не заменяет итоговое медицинское заключение."
    )
