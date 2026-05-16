import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATIENT_ID = "369710"
CSV_PATH = ROOT / "data" / "train1.csv"


def _print_row(fieldnames: list[str], row: dict[str, str], width: int, label: str, path: Path) -> None:
    for key in fieldnames:
        val = row.get(key, "") or ""
        if key == "specialist_conclusions" and val.strip():
            print(f"{key}:")
            try:
                parsed = json.loads(val)
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(val)
            print()
            continue
        print(f"{key:{width}}  {val}")
    print(f"({label} из {path})")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None) or ""
        if enc.lower() != "utf-8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass
    want = str(PATIENT_ID).strip()
    if not want:
        print("PATIENT_ID is empty", file=sys.stderr)
        sys.exit(1)
    path = CSV_PATH
    if not path.is_file():
        print(f"file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("empty csv", file=sys.stderr)
            sys.exit(1)
        if "patient_id" not in reader.fieldnames:
            print("csv has no patient_id column", file=sys.stderr)
            sys.exit(1)
        matches: list[dict[str, str]] = []
        for r in reader:
            pid = (r.get("patient_id") or "").strip()
            if pid == want:
                matches.append(r)
    if not matches:
        print(f"нет записей с patient_id={want!r}", file=sys.stderr)
        sys.exit(1)
    width = max(len(k) for k in reader.fieldnames)
    for idx, row in enumerate(matches, start=1):
        if len(matches) > 1:
            print(f"=== match {idx}/{len(matches)} ===")
        eid = (row.get("exam_row_id") or "").strip() or str(idx)
        _print_row(reader.fieldnames, row, width, f"patient_id={want}, exam_row_id={eid}", path)
        if idx < len(matches):
            print()


if __name__ == "__main__":
    main()
