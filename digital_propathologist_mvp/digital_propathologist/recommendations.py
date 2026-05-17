from __future__ import annotations

import re

from .types import Recommendation, ResearchWarning, SpecialistConclusion

RECOMMENDATION_PATTERNS = [
    r"[^.!?\n]*(?:годен|годна)\s+в\s+очках[^.!?\n]*",
    r"[^.!?\n]*работ[аы]?\s+в\s+очках[^.!?\n]*",
    r"[^.!?\n]*(?:рекомендовано|рекомендована|рекомендован|рекомендац)[^.!?\n]*",
    r"[^.!?\n]*(?:наблюдение|контроль|целевое\s+АД|санац|дообслед|консультац|лечение|повторный\s+осмотр)[^.!?\n]*",
    r"[^.!?\n]*нуждается[^.!?\n]*(?:санац|дообслед|консультац|лечен)[^.!?\n]*",
]

WARNING_PATTERNS = [
    ("глюкоза", r"глюкоз|сахар"),
    ("холестерин", r"холест"),
    ("артериальное давление", r"\bАД\b|гипертенз|гипертонич|артериальн"),
    ("ЭКГ", r"тахикард|реполяризац|экстрасистол|блокад|нарушен.*ритм|ЧСС\s*[>=]\s*9|ЧСС\s*[1-9][0-9]{2}"),
    ("слух", r"тугоух|снижен.*слух|нейросенсор|сенсоневраль|H90"),
    ("зрение", r"миоп|астигмат|катаракт|амблиоп|H52|H53|очки"),
    ("стоматология", r"санац|кариес|зубн"),
    ("неврология", r"остеохондроз|дорсопат|M42|радикул"),
    ("дыхательная система", r"астма|ХОБЛ|бронхит|спирометр|обструкц"),
]

RESEARCH_SPECIALISTS = [
    "аудиометр",
    "спирометр",
    "электрокардиограф",
    "электроэнцеф",
    "энцефалограм",
    "ээг",
    "экг",
    "рентген",
    "флюорограф",
    "узи",
    "анализ",
    "лаборатор",
    "глюкоз",
    "холестерин",
]


def recommendation_kind(text: str, source: str) -> str:
    low = f"{text} {source}".lower()
    if "очк" in low or "зрен" in low or "миоп" in low:
        return "зрение"
    if "санац" in low or "кариес" in low or "зуб" in low:
        return "стоматология"
    if "ад" in low or "гипертенз" in low or "контроль" in low:
        return "терапия"
    if "дообслед" in low or "консультац" in low:
        return "дообследование"
    return "прочее"


def extract_recommendations(conclusions: list[SpecialistConclusion]) -> list[Recommendation]:
    result: list[Recommendation] = []
    seen: set[tuple[str, str]] = set()
    for item in conclusions:
        text = item.conclusion or ""
        if not text:
            continue
        for pattern in RECOMMENDATION_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = re.sub(r"\s+", " ", match.group(0)).strip(" .;,-")
                if len(value) < 5:
                    continue
                key = (item.specialist, value.lower())
                if key in seen:
                    continue
                seen.add(key)
                result.append(
                    Recommendation(
                        source=item.specialist or "не указан",
                        text=value,
                        kind=recommendation_kind(value, item.specialist),
                    )
                )
    return result


def is_research(item: SpecialistConclusion) -> bool:
    low = f"{item.specialist} {item.conclusion} {item.mkb_description}".lower()
    return any(word in low for word in RESEARCH_SPECIALISTS)


def detect_research_warnings(conclusions: list[SpecialistConclusion]) -> list[ResearchWarning]:
    warnings: list[ResearchWarning] = []
    for item in conclusions:
        blob = item.text_blob
        if not blob:
            continue
        for kind, pattern in WARNING_PATTERNS:
            if re.search(pattern, blob, flags=re.IGNORECASE):
                text = item.conclusion or item.mkb_description or item.mkb_code or "Настораживающий признак найден в заключении."
                warnings.append(
                    ResearchWarning(
                        source=item.specialist or "не указан",
                        text=re.sub(r"\s+", " ", text).strip(),
                        kind=kind,
                        severity="warning",
                    )
                )
                break
    return warnings
