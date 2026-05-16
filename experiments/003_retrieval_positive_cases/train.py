import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import (  # noqa: E402
    autonomous_score,
    build_feature_text,
    labels_to_output,
    prepare_train_df,
    split_factors,
    split_train_valid,
    write_submission,
)


def threshold_values(spec: list[float]) -> list[float]:
    start, stop, step = spec
    return [round(float(x), 4) for x in np.arange(start, stop + step / 2, step)]


def make_vectorizer(params: dict) -> TfidfVectorizer:
    vectorizer_params = dict(params["vectorizer"])
    vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    return TfidfVectorizer(**vectorizer_params)


def feature_series(df: pd.DataFrame, params: dict) -> pd.Series:
    return df.apply(
        build_feature_text,
        axis=1,
        repeat_assigned=int(params["features"].get("repeat_assigned", 4)),
        include_month=bool(params["features"].get("include_month", False)),
    )


def top_neighbors(query_matrix, positive_matrix, top_k: int):
    sims = query_matrix @ positive_matrix.T
    neighbors: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz == 0:
            neighbors.append((np.array([], dtype=int), np.array([], dtype=float)))
            continue
        if row.nnz > top_k:
            pick = np.argpartition(row.data, -top_k)[-top_k:]
        else:
            pick = np.arange(row.nnz)
        order = pick[np.argsort(row.data[pick])[::-1]]
        neighbors.append((row.indices[order], row.data[order]))
    return neighbors


def retrieval_predict(
    neighbors: list[tuple[np.ndarray, np.ndarray]],
    query_assigned: list[list[str]],
    positive_assigned: list[set[str]],
    positive_labels: list[list[str]],
    factor_prior: dict[str, float],
    *,
    any_threshold: float,
    factor_threshold: float,
    max_factors: int,
    same_factor_bonus: float,
    prior_weight: float,
    min_shared_assigned: int,
) -> list[list[str]]:
    predictions: list[list[str]] = []

    for (idxs, scores), assigned in zip(neighbors, query_assigned):
        assigned_set = set(assigned)
        factor_scores: dict[str, float] = defaultdict(float)

        for neighbor_idx, sim in zip(idxs, scores):
            shared = assigned_set & positive_assigned[neighbor_idx]
            if len(shared) < min_shared_assigned:
                continue
            weight = float(sim) * (1.0 + same_factor_bonus * len(shared))
            for factor in positive_labels[neighbor_idx]:
                if factor in assigned_set:
                    factor_scores[factor] += weight + prior_weight * factor_prior.get(factor, 0.0)

        total_score = sum(factor_scores.values())
        if total_score < any_threshold or not factor_scores:
            predictions.append([])
            continue

        selected = [
            factor for factor, score in sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
            if score >= factor_threshold
        ][:max_factors]
        predictions.append(selected)

    return predictions


def update_info(metrics: dict, submission_path: Path) -> None:
    info_path = EXPERIMENT_DIR / "info.yml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    info["status"] = "done"
    info["local_validation"] = {
        "autonomous_score_alpha_0_5": metrics["best_metrics"]["autonomous_score"],
        "any_f1": metrics["best_metrics"]["any_f1"],
        "jaccard": metrics["best_metrics"]["factor_jaccard_on_true_positive_rows"],
        "positive_rate_pred": metrics["best_metrics"]["positive_rate_pred"],
        "positive_rate_true": metrics["best_metrics"]["positive_rate_true"],
    }
    info["outputs"] = {
        "submission": str(submission_path.relative_to(ROOT)),
        "validation_predictions": "experiments/003_retrieval_positive_cases/validation_predictions.csv",
        "metrics": "experiments/003_retrieval_positive_cases/metrics.json",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    alpha = float(params["metric"]["alpha"])
    train_df = prepare_train_df(ROOT / params["data"]["train"])
    test_df = pd.read_csv(ROOT / params["data"]["test"])
    train_part, val_part = split_train_valid(train_df, params)

    pos_part = train_part[train_part["target_any"]].reset_index(drop=True)
    vectorizer = make_vectorizer(params)
    x_pos = vectorizer.fit_transform(feature_series(pos_part, params))
    x_val = vectorizer.transform(feature_series(val_part, params))

    positive_assigned = pos_part["assigned_harmful_factors"].apply(lambda x: set(split_factors(x))).tolist()
    positive_labels = pos_part["target_factors"].tolist()
    query_assigned_val = val_part["assigned_harmful_factors"].apply(split_factors).tolist()
    true_labels = val_part["target_factors"].tolist()

    label_counts = train_part["target_factors"].explode().dropna().value_counts()
    assigned_counts = train_part["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    factor_prior = {
        factor: float(label_counts.get(factor, 0) / max(assigned_counts.get(factor, 1), 1))
        for factor in sorted(label_counts.index.astype(str))
    }

    grid = params["retrieval"]
    neighbor_cache = {
        int(k): top_neighbors(x_val, x_pos, int(k))
        for k in grid["top_k_grid"]
    }

    best = None
    tried = 0
    for top_k, neighbors in neighbor_cache.items():
        for any_threshold in threshold_values(grid["any_threshold_grid"]):
            for factor_threshold in threshold_values(grid["factor_threshold_grid"]):
                for max_factors in grid["max_factors_grid"]:
                    for same_factor_bonus in grid["same_factor_bonus_grid"]:
                        for prior_weight in grid["prior_weight_grid"]:
                            for min_shared_assigned in grid["min_shared_assigned_grid"]:
                                pred = retrieval_predict(
                                    neighbors,
                                    query_assigned_val,
                                    positive_assigned,
                                    positive_labels,
                                    factor_prior,
                                    any_threshold=any_threshold,
                                    factor_threshold=factor_threshold,
                                    max_factors=int(max_factors),
                                    same_factor_bonus=float(same_factor_bonus),
                                    prior_weight=float(prior_weight),
                                    min_shared_assigned=int(min_shared_assigned),
                                )
                                metrics = autonomous_score(true_labels, pred, alpha=alpha)
                                tried += 1
                                if best is None or metrics["autonomous_score"] > best["metrics"]["autonomous_score"]:
                                    best = {
                                        "top_k": top_k,
                                        "any_threshold": any_threshold,
                                        "factor_threshold": factor_threshold,
                                        "max_factors": int(max_factors),
                                        "same_factor_bonus": float(same_factor_bonus),
                                        "prior_weight": float(prior_weight),
                                        "min_shared_assigned": int(min_shared_assigned),
                                        "metrics": metrics,
                                        "pred": pred,
                                    }
                                    print(
                                        "new_best",
                                        json.dumps({k: v for k, v in best.items() if k not in {"pred"}}, ensure_ascii=False),
                                        flush=True,
                                    )

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    pos_all = train_df[train_df["target_any"]].reset_index(drop=True)
    vectorizer = make_vectorizer(params)
    x_pos_all = vectorizer.fit_transform(feature_series(pos_all, params))
    x_test = vectorizer.transform(feature_series(test_df, params))
    positive_assigned_all = pos_all["assigned_harmful_factors"].apply(lambda x: set(split_factors(x))).tolist()
    positive_labels_all = pos_all["target_factors"].tolist()

    label_counts_all = train_df["target_factors"].explode().dropna().value_counts()
    assigned_counts_all = train_df["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    factor_prior_all = {
        factor: float(label_counts_all.get(factor, 0) / max(assigned_counts_all.get(factor, 1), 1))
        for factor in sorted(label_counts_all.index.astype(str))
    }

    test_neighbors = top_neighbors(x_test, x_pos_all, best["top_k"])
    test_pred = retrieval_predict(
        test_neighbors,
        test_df["assigned_harmful_factors"].apply(split_factors).tolist(),
        positive_assigned_all,
        positive_labels_all,
        factor_prior_all,
        any_threshold=best["any_threshold"],
        factor_threshold=best["factor_threshold"],
        max_factors=best["max_factors"],
        same_factor_bonus=best["same_factor_bonus"],
        prior_weight=best["prior_weight"],
        min_shared_assigned=best["min_shared_assigned"],
    )

    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], test_pred)

    metrics = {
        "experiment": params["name"],
        "rows_train_total": int(len(train_df)),
        "rows_positive_fit": int(len(pos_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "target_positive_rate": round(float(train_df["target_any"].mean()), 6),
        "best_params": {k: v for k, v in best.items() if k not in {"metrics", "pred"}},
        "best_metrics": best["metrics"],
        "grid_combinations_tried": tried,
        "submission_positive_rows": int(sum(bool(x) for x in test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in test_pred])), 6),
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_info(metrics, submission_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
