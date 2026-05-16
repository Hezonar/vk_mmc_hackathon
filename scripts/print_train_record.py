import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 1
CSV_PATH = ROOT / "data" / "train1.csv"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", None) or ""
        if enc.lower() != "utf-8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except OSError:
                pass
    if RECORD < 1:
        print("RECORD must be >= 1", file=sys.stderr)
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
        row = None
        for i, r in enumerate(reader, start=1):
            if i == RECORD:
                row = r
                break
        if row is None:
            print(f"нет записи с номером {RECORD}", file=sys.stderr)
            sys.exit(1)
    width = max(len(k) for k in reader.fieldnames)
    for key in reader.fieldnames:
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
    print(f"(запись {RECORD} из {path})")


if __name__ == "__main__":
    main()
