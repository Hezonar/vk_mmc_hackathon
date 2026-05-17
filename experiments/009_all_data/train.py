import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import prepare_train_df, write_submission  # noqa: E402


def load_exp006():
    path = ROOT / "experiments" / "006_candidate_factor_binary" / "train.py"
    spec = importlib.util.spec_from_file_location("exp006_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def vectorizer_bundle(vectorizer) -> dict:
    return {
        "word": vectorizer.word,
        "char": vectorizer.char,
    }


def predict_candidates(exp006, artifact: dict, df: pd.DataFrame) -> list[list[str]]:
    candidates = exp006.make_candidate_frame(df, artifact["params"], has_target=False)
    if candidates.empty:
        return [[] for _ in range(len(df))]
    texts = candidates["text"].tolist()
    x = hstack([
        artifact["vectorizer"]["word"].transform(texts),
        artifact["vectorizer"]["char"].transform(texts),
    ], format="csr")
    scores = artifact["model"].predict_proba(x)[:, 1]
    threshold = float(artifact["threshold"])

    selected_by_row: dict[object, list[tuple[str, float]]] = {}
    for (_, row), score in zip(candidates.iterrows(), scores):
        if score >= threshold:
            selected_by_row.setdefault(row["source_index"], []).append((row["factor"], float(score)))

    predictions = []
    for row_idx in df.index:
        selected = sorted(selected_by_row.get(row_idx, []), key=lambda x: x[1], reverse=True)
        predictions.append([factor for factor, _ in selected])
    return predictions


def update_info(summary: dict) -> None:
    info_path = EXPERIMENT_DIR / "info.yml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    info["status"] = "done"
    info["outputs"] = {
        "artifact": summary["artifact_path"],
        "submission": "experiments/009_all_data/submission.csv",
        "summary": "experiments/009_all_data/summary.json",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    exp006 = load_exp006()
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    train_df = prepare_train_df(ROOT / params["data"]["train"])
    test_df = pd.read_csv(ROOT / params["data"]["test"])

    train_candidates = exp006.make_candidate_frame(train_df, params, has_target=True)
    vectorizer = exp006.DualTfidf(params)
    x_train = vectorizer.fit_transform(train_candidates["text"].tolist())
    model = LogisticRegression(**params["model"])
    model.fit(x_train, train_candidates["target"].astype(int))

    artifact = {
        "model_version": "exp009_all_data_candidate_binary",
        "source_experiment": params["source_experiment"],
        "params": params,
        "threshold": float(params["prediction"]["factor_threshold"]),
        "vectorizer": vectorizer_bundle(vectorizer),
        "model": model,
        "candidate_positive_rate_train": float(train_candidates["target"].mean()),
        "train_rows": int(len(train_df)),
        "candidate_rows": int(len(train_candidates)),
    }

    artifact_path = ROOT / params["artifact"]["model_path"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path, compress=3)

    test_pred = predict_candidates(exp006, artifact, test_df)
    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], test_pred)

    summary = {
        "experiment": params["name"],
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "candidate_rows_train": int(len(train_candidates)),
        "candidate_positive_rate_train": round(float(train_candidates["target"].mean()), 6),
        "threshold": float(params["prediction"]["factor_threshold"]),
        "submission_positive_rows": int(sum(bool(x) for x in test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in test_pred])), 6),
    }
    (EXPERIMENT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    update_info(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
