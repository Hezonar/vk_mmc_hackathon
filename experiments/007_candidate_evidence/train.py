import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import (  # noqa: E402
    autonomous_score,
    build_feature_text,
    clean_token,
    labels_to_output,
    prepare_train_df,
    split_factors,
    write_submission,
)


def range_values(spec: list[float]) -> list[float]:
    start, stop, step = spec
    return [round(float(x), 4) for x in np.arange(start, stop + step / 2, step)]


def vectorizer_params(params: dict, key: str) -> dict:
    out = dict(params[key])
    if "ngram_range" in out:
        out["ngram_range"] = tuple(out["ngram_range"])
    return out


def factor_family(factor: str) -> str:
    return re.split(r"[.]", factor.strip())[0]


def base_text(row: pd.Series, params: dict) -> str:
    return build_feature_text(
        row,
        repeat_assigned=int(params["features"]["repeat_assigned"]),
        include_month=bool(params["features"]["include_month"]),
    )


def candidate_text(row: pd.Series, factor: str, params: dict) -> str:
    candidate = clean_token(factor, "candidate_factor")
    family = clean_token(factor_family(factor), "candidate_family")
    pieces = [base_text(row, params)]
    pieces.extend([candidate] * int(params["features"]["candidate_repeats"]))
    pieces.extend([family] * max(1, int(params["features"]["candidate_repeats"]) // 2))
    pieces.extend(["candidate_pair_" + candidate] * int(params["features"]["candidate_pair_repeats"]))
    return " ".join(pieces)


def make_candidate_frame(df: pd.DataFrame, params: dict, *, has_target: bool) -> pd.DataFrame:
    rows = []
    for row_idx, row in df.iterrows():
        assigned = split_factors(row.get("assigned_harmful_factors", ""))
        target = set(row.get("target_factors", [])) if has_target else set()
        assigned_count = len(assigned)
        for factor in assigned:
            rows.append({
                "source_index": row_idx,
                "exam_row_id": row["exam_row_id"],
                "factor": factor,
                "factor_family": factor_family(factor),
                "assigned_count": assigned_count,
                "text": candidate_text(row, factor, params),
                "base_text": base_text(row, params),
                "target": int(factor in target) if has_target else 0,
            })
    return pd.DataFrame(rows)


def factor_stats(train_candidates: pd.DataFrame) -> dict[str, dict[str, float]]:
    grouped = train_candidates.groupby("factor")["target"]
    total = grouped.count()
    pos = grouped.sum()
    stats = {}
    global_rate = float(train_candidates["target"].mean())
    for factor in total.index:
        # Laplace smoothing matters for rare factors.
        stats[factor] = {
            "factor_count": float(total[factor]),
            "factor_pos": float(pos[factor]),
            "factor_rate": float((pos[factor] + 1.0) / (total[factor] + 2.0)),
            "factor_log_count": float(np.log1p(total[factor])),
            "global_rate": global_rate,
        }
    return stats


class EvidenceBuilder:
    def __init__(self, params: dict):
        self.params = params
        self.vectorizer = TfidfVectorizer(**vectorizer_params(params, "evidence_vectorizer"))
        self.positive_by_factor: dict[str, csr_matrix] = {}

    def fit(self, train_candidates: pd.DataFrame) -> None:
        self.vectorizer.fit(train_candidates["base_text"].tolist())
        positive = train_candidates[train_candidates["target"] == 1]
        for factor, group in positive.groupby("factor"):
            self.positive_by_factor[factor] = self.vectorizer.transform(group["base_text"].tolist())

    def transform(self, candidates: pd.DataFrame) -> np.ndarray:
        base_matrix = self.vectorizer.transform(candidates["base_text"].tolist())
        top_k = int(self.params["retrieval"]["top_k_same_factor"])
        features = np.zeros((len(candidates), 4), dtype=np.float32)
        factor_to_rows: dict[str, list[int]] = defaultdict(list)
        for i, factor in enumerate(candidates["factor"].tolist()):
            factor_to_rows[factor].append(i)

        for factor, rows in factor_to_rows.items():
            pos_matrix = self.positive_by_factor.get(factor)
            if pos_matrix is None or pos_matrix.shape[0] == 0:
                continue
            sims = base_matrix[rows] @ pos_matrix.T
            for local_i, global_i in enumerate(rows):
                row = sims.getrow(local_i)
                if row.nnz == 0:
                    continue
                data = row.data
                if len(data) > top_k:
                    data = data[np.argpartition(data, -top_k)[-top_k:]]
                features[global_i, 0] = float(np.max(data))
                features[global_i, 1] = float(np.mean(data))
                features[global_i, 2] = float(np.sum(data))
                features[global_i, 3] = float(len(data))
        return features


class FeatureUnion:
    def __init__(self, params: dict):
        self.params = params
        self.word = TfidfVectorizer(**vectorizer_params(params, "word_vectorizer"))
        self.char = TfidfVectorizer(**vectorizer_params(params, "char_vectorizer"))
        self.evidence = EvidenceBuilder(params)
        self.scaler = StandardScaler()
        self.stats: dict[str, dict[str, float]] = {}

    def numeric_features(self, candidates: pd.DataFrame, retrieval_features: np.ndarray) -> np.ndarray:
        rows = []
        for _, row in candidates.iterrows():
            stats = self.stats.get(row["factor"], {
                "factor_count": 0.0,
                "factor_pos": 0.0,
                "factor_rate": 0.0,
                "factor_log_count": 0.0,
                "global_rate": 0.0,
            })
            rows.append([
                float(row["assigned_count"]),
                float(stats["factor_count"]),
                float(stats["factor_pos"]),
                float(stats["factor_rate"]),
                float(stats["factor_log_count"]),
                float(stats["global_rate"]),
                *retrieval_features[len(rows)].tolist(),
            ])
        return np.array(rows, dtype=np.float32)

    def fit_transform(self, candidates: pd.DataFrame):
        self.stats = factor_stats(candidates)
        self.evidence.fit(candidates)
        retrieval = self.evidence.transform(candidates)
        numeric = self.scaler.fit_transform(self.numeric_features(candidates, retrieval))
        return hstack([
            self.word.fit_transform(candidates["text"].tolist()),
            self.char.fit_transform(candidates["text"].tolist()),
            csr_matrix(numeric),
        ], format="csr")

    def transform(self, candidates: pd.DataFrame):
        retrieval = self.evidence.transform(candidates)
        numeric = self.scaler.transform(self.numeric_features(candidates, retrieval))
        return hstack([
            self.word.transform(candidates["text"].tolist()),
            self.char.transform(candidates["text"].tolist()),
            csr_matrix(numeric),
        ], format="csr")


def rows_from_candidates(candidate_df: pd.DataFrame, scores: np.ndarray, threshold: float, row_order: pd.DataFrame) -> list[list[str]]:
    selected_by_row: dict[object, list[tuple[str, float]]] = {}
    for (_, row), score in zip(candidate_df.iterrows(), scores):
        if score >= threshold:
            selected_by_row.setdefault(row["source_index"], []).append((row["factor"], float(score)))

    predictions = []
    for row_idx in row_order.index:
        selected = sorted(selected_by_row.get(row_idx, []), key=lambda x: x[1], reverse=True)
        predictions.append([factor for factor, _ in selected])
    return predictions


def objective(metrics: dict, params: dict) -> float:
    cfg = params["prediction"]
    pred_rate = metrics["positive_rate_pred"]
    penalty = 0.0
    if pred_rate > float(cfg["positive_rate_soft_max"]):
        penalty += float(cfg["positive_rate_penalty"]) * (pred_rate - float(cfg["positive_rate_soft_max"]))
    if pred_rate < float(cfg["positive_rate_soft_min"]):
        penalty += float(cfg["positive_rate_penalty"]) * (float(cfg["positive_rate_soft_min"]) - pred_rate)
    target_penalty = 0.30 * abs(pred_rate - float(cfg["target_positive_rate"]))
    return (
        metrics["autonomous_score"]
        + float(cfg["jaccard_weight_bonus"]) * metrics["factor_jaccard_on_true_positive_rows"]
        - penalty
        - target_penalty
    )


def update_info(metrics: dict, submission_path: Path) -> None:
    info_path = EXPERIMENT_DIR / "info.yml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    best = metrics["best_metrics"]
    info["status"] = "done"
    info["local_validation"] = {
        "autonomous_score_alpha_0_5": best["autonomous_score"],
        "objective": metrics["best_objective"],
        "any_f1": best["any_f1"],
        "jaccard": best["factor_jaccard_on_true_positive_rows"],
        "positive_rate_pred": best["positive_rate_pred"],
        "positive_rate_true": best["positive_rate_true"],
    }
    info["outputs"] = {
        "submission": str(submission_path.relative_to(ROOT)),
        "validation_predictions": "experiments/007_candidate_evidence/validation_predictions.csv",
        "metrics": "experiments/007_candidate_evidence/metrics.json",
        "validation_candidates": "experiments/007_candidate_evidence/validation_candidates.csv",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    train_df = prepare_train_df(ROOT / params["data"]["train"])
    test_df = pd.read_csv(ROOT / params["data"]["test"])

    train_part, val_part = train_test_split(
        train_df,
        train_size=float(params["split"]["train_size"]),
        random_state=int(params["split"]["random_state"]),
        stratify=train_df["target_any"],
    )

    train_candidates = make_candidate_frame(train_part, params, has_target=True)
    val_candidates = make_candidate_frame(val_part, params, has_target=True)

    features = FeatureUnion(params)
    x_train = features.fit_transform(train_candidates)
    model = LogisticRegression(**params["model"])
    model.fit(x_train, train_candidates["target"].astype(int))

    x_val = features.transform(val_candidates)
    val_scores = model.predict_proba(x_val)[:, 1]
    val_candidates_out = val_candidates[["exam_row_id", "factor", "target"]].copy()
    val_candidates_out["score"] = val_scores
    val_candidates_out.to_csv(EXPERIMENT_DIR / "validation_candidates.csv", index=False)

    true_labels = val_part["target_factors"].tolist()
    best = None
    tried = 0
    for threshold in range_values(params["prediction"]["factor_threshold_grid"]):
        pred = rows_from_candidates(val_candidates, val_scores, threshold, val_part)
        metrics = autonomous_score(true_labels, pred, alpha=float(params["metric"]["alpha"]))
        obj = objective(metrics, params)
        tried += 1
        if best is None or obj > best["objective"]:
            best = {
                "objective": round(float(obj), 6),
                "params": {"factor_threshold": threshold},
                "metrics": metrics,
                "pred": pred,
            }
            print("new_best", json.dumps({k: v for k, v in best.items() if k != "pred"}, ensure_ascii=False), flush=True)

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    all_candidates = make_candidate_frame(train_df, params, has_target=True)
    features = FeatureUnion(params)
    x_all = features.fit_transform(all_candidates)
    model = LogisticRegression(**params["model"])
    model.fit(x_all, all_candidates["target"].astype(int))

    test_candidates = make_candidate_frame(test_df, params, has_target=False)
    x_test = features.transform(test_candidates)
    test_scores = model.predict_proba(x_test)[:, 1]
    test_pred = rows_from_candidates(test_candidates, test_scores, best["params"]["factor_threshold"], test_df)

    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], test_pred)

    metrics = {
        "experiment": params["name"],
        "rows_train_total": int(len(train_df)),
        "rows_train_fit": int(len(train_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "candidate_rows_train_fit": int(len(train_candidates)),
        "candidate_rows_validation": int(len(val_candidates)),
        "candidate_positive_rate_train_fit": round(float(train_candidates["target"].mean()), 6),
        "target_positive_rate_total": round(float(train_df["target_any"].mean()), 6),
        "best_objective": best["objective"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "grid_combinations_tried": tried,
        "submission_positive_rows": int(sum(bool(x) for x in test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in test_pred])), 6),
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_info(metrics, submission_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
