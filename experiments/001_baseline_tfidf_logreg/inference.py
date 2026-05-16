import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    src = EXP_DIR / "submission.csv"
    if not src.exists():
        src = ROOT / "reports" / "ml_baseline" / "submission_baseline.csv"
    shutil.copyfile(src, ROOT / "submission.csv")
    print(f"Copied {src} -> {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()
