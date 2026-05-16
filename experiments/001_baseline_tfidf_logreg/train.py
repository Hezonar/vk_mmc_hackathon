import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


def main() -> None:
    source = ROOT / "reports" / "ml_baseline" / "submission_baseline.csv"
    target = EXP_DIR / "submission.csv"
    if not source.exists():
        raise SystemExit("Original baseline output is missing. Run scripts/train_factor_baseline.py first.")
    shutil.copyfile(source, target)
    print(f"Copied historical baseline submission to {target}")


if __name__ == "__main__":
    main()
