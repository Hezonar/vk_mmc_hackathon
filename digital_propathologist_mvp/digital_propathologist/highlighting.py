from __future__ import annotations

import re

from .types import Exam, PredictionResult, SpecialistConclusion

HEALTHY_MARKERS = [
    "z00.0",
    "практически здоров",
    "здоров",
    "патологии не выявлено",
    "кожные заболевания не выявлены",
    "хирургической патологии не выявлено",
]

EMPTY_MARKERS = ["", "nan", "none", "null"]


LINK_RULES: list[tuple[str, str]] = [
    ("4.4", r"аудиометр|оториноларинголог|лор|слух|тугоух|нейросенсор|сенсоневраль|H90"),
    ("6", r"терапевт|гипертенз|гипертонич|\bI10\b|I11|АД|сердеч|кардио|невролог|остеохондроз|M42"),
    ("6.1", r"терапевт|гипертенз|гипертонич|\bI10\b|I11|АД|сердеч|кардио|невролог|остеохондроз|M42"),
    ("6.2", r"терапевт|гипертенз|гипертонич|\bI10\b|I11|АД|сердеч|кардио|невролог|остеохондроз|M42"),
    ("15", r"офтальмолог|зрен|миоп|астигмат|амблиоп|катаракт|очки|H52|H53|Z96"),
    ("14", r"офтальмолог|зрен|миоп|астигмат|амблиоп|катаракт|очки|H52|H53|Z96"),
    ("12", r"стоматолог|кариес|санац|терапевт|астма|бронх|спирометр|дых"),
    ("13", r"терапевт|астма|бронх|спирометр|дых|хирург|грыж"),
    ("17", r"терапевт|астма|бронх|спирометр|дых|аллерг|дермат"),
]

WARNING_TEXT_RULE = r"гипертенз|гипертонич|тугоух|снижен.*слух|миоп|астигмат|катаракт|амблиоп|остеохондроз|кариес|санац|астма|ожирение|энцефалопат|реполяризац|тахикард|нуждается|годен в очках"


def normalize_factor(factor: str) -> str:
    return str(factor).strip()


def text_blob(item: SpecialistConclusion) -> str:
    return item.text_blob


def card_has_warning(item: SpecialistConclusion) -> bool:
    if not item.has_meaningful_result:
        return False
    blob = text_blob(item).lower()
    if not blob.strip():
        return False
    if re.search(WARNING_TEXT_RULE, blob, flags=re.IGNORECASE):
        return True
    if item.mkb_code and item.mkb_code.strip() and item.mkb_code.strip().upper() != "Z00.0":
        return True
    return False


def card_is_empty(item: SpecialistConclusion) -> bool:
    return not item.has_meaningful_result


def card_is_healthy(item: SpecialistConclusion) -> bool:
    blob = text_blob(item).lower()
    return any(marker in blob for marker in HEALTHY_MARKERS) and not card_has_warning(item)


def build_highlight_links(exam: Exam, prediction: PredictionResult | None = None) -> dict[str, list[int]]:
    links: dict[str, list[int]] = {factor: [] for factor in exam.assigned_harmful_factors}
    factors_to_consider = exam.assigned_harmful_factors
    if prediction and prediction.factors:
        factors_to_consider = list(set(exam.assigned_harmful_factors + prediction.factors))

    for factor in factors_to_consider:
        for idx, item in enumerate(exam.specialist_conclusions):
            if not item.has_meaningful_result:
                continue
            blob = text_blob(item)
            for rule_factor, pattern in LINK_RULES:
                exact = factor == rule_factor
                prefix = rule_factor in {"6", "6.1", "6.2"} and factor.startswith("6")
                if (exact or prefix) and re.search(pattern, blob, flags=re.IGNORECASE):
                    links.setdefault(factor, [])
                    if idx not in links[factor]:
                        links[factor].append(idx)
    return links


def factor_status(factor: str, exam: Exam, prediction: PredictionResult | None) -> str:
    if not prediction or not prediction.done:
        return "gray"
    if prediction.status == "insufficient":
        return "blue"
    if factor in prediction.factors:
        return "red"
    links = prediction.linked_factors or {}
    # Yellow means there are suspicious related cards but the factor is not in final predicted factors.
    for idx in links.get(factor, []):
        if 0 <= idx < len(exam.specialist_conclusions) and card_has_warning(exam.specialist_conclusions[idx]):
            return "yellow"
    return "green"


def card_status(idx: int, item: SpecialistConclusion, prediction: PredictionResult | None) -> str:
    if card_is_empty(item):
        return "empty"
    if prediction and prediction.done:
        for factor, indexes in (prediction.linked_factors or {}).items():
            if factor in prediction.factors and idx in indexes:
                return "risk"
    if card_has_warning(item):
        return "warning"
    return "normal"


def linked_factors_for_card(idx: int, links: dict[str, list[int]]) -> list[str]:
    return [factor for factor, indexes in links.items() if idx in indexes]
