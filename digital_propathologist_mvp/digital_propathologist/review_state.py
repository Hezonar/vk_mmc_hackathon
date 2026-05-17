from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = ROOT / "data" / "review_state.json"
DAILY_TARGET = 50


def today_iso() -> str:
    return date.today().isoformat()


def empty_state(day: str | None = None) -> dict[str, Any]:
    return {
        "current_date": day or today_iso(),
        "daily_target": DAILY_TARGET,
        "reviewed_today": [],
        "history": [],
    }


def load_review_state(path: str | Path = DEFAULT_STATE_PATH, *, day: str | None = None) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return empty_state(day)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty_state(day)

    current_day = day or today_iso()
    state.setdefault("daily_target", DAILY_TARGET)
    state.setdefault("reviewed_today", [])
    state.setdefault("history", [])
    if state.get("current_date") != current_day:
        state["history"].extend(state.get("reviewed_today", []))
        state["reviewed_today"] = []
        state["current_date"] = current_day
    return state


def save_review_state(state: dict[str, Any], path: str | Path = DEFAULT_STATE_PATH) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def review_key(patient_id: str, exam_row_id: str) -> str:
    return f"{patient_id.strip()}::{exam_row_id.strip()}"


def is_reviewed(state: dict[str, Any], patient_id: str, exam_row_id: str) -> bool:
    key = review_key(patient_id, exam_row_id)
    return any(item.get("key") == key for item in state.get("reviewed_today", []))


def mark_reviewed(
    patient_id: str,
    exam_row_id: str,
    *,
    path: str | Path = DEFAULT_STATE_PATH,
    day: str | None = None,
) -> tuple[dict[str, Any], bool]:
    state = load_review_state(path, day=day)
    key = review_key(patient_id, exam_row_id)
    if is_reviewed(state, patient_id, exam_row_id):
        return state, False

    state["reviewed_today"].append(
        {
            "key": key,
            "patient_id": patient_id,
            "exam_row_id": exam_row_id,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_review_state(state, path)
    return state, True


def reviewed_count(state: dict[str, Any]) -> int:
    return len(state.get("reviewed_today", []))
