import shutil
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT))


def main() -> None:
    src = EXPERIMENT_DIR / "submission.csv"
    if not src.exists():
        raise SystemExit("Run train.py first: submission.csv is missing.")
    dst = ROOT / "submission.csv"
    shutil.copyfile(src, dst)
    print(f"Copied {src} -> {dst}")


if __name__ == "__main__":
    main()
