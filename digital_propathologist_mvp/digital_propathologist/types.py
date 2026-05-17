from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FactorColor = Literal["gray", "green", "red", "yellow", "blue"]
CardStatus = Literal["normal", "warning", "risk", "empty"]
MlVerdict = Literal["fit", "not_fit", "insufficient", "not_run"]


@dataclass
class SpecialistConclusion:
    specialist: str = ""
    consultation_date: str = ""
    conclusion: str = ""
    health_group: str = ""
    mkb_code: str = ""
    mkb_description: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def text_blob(self) -> str:
        return " ".join(
            [
                self.specialist or "",
                self.conclusion or "",
                self.health_group or "",
                self.mkb_code or "",
                self.mkb_description or "",
            ]
        ).strip()

    @property
    def has_meaningful_result(self) -> bool:
        """True when a doctor/research card contains an actual result."""
        if (self.conclusion or "").strip():
            return True
        if (self.health_group or "").strip():
            return True

        mkb_code = (self.mkb_code or "").strip().upper()
        mkb_description = (self.mkb_description or "").strip().lower()
        if mkb_code:
            return True
        return bool(mkb_description and mkb_description != "общий медицинский осмотр")


@dataclass
class Exam:
    exam_row_id: str
    patient_id: str
    medical_exam_id: str = ""
    consultation_date: str = ""
    assigned_harmful_factors: list[str] = field(default_factory=list)
    contraindicated_factors: list[str] = field(default_factory=list)
    has_contraindications: bool | None = None
    specialist_conclusions: list[SpecialistConclusion] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    source: str
    text: str
    kind: str = "прочее"


@dataclass
class ResearchWarning:
    source: str
    text: str
    kind: str = "настораживающий признак"
    severity: CardStatus = "warning"


@dataclass
class PredictionResult:
    done: bool = False
    loading: bool = False
    factors: list[str] = field(default_factory=list)
    status: Literal["not_run", "ok", "risk", "insufficient"] = "not_run"
    explanation: str = "Предсказание еще не выполнено."
    ml_verdict: MlVerdict = "not_run"
    model_version: str = ""
    threshold: float | None = None
    factor_scores: list[dict[str, Any]] = field(default_factory=list)
    linked_factors: dict[str, list[int]] = field(default_factory=dict)

    @property
    def factors_csv(self) -> str:
        return ";".join(self.factors) if self.factors else "0"
