import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))
sys.path.append(str(ROOT / "experiments" / "003_retrieval_positive_cases"))

from medical_factor_baseline import (  # noqa: E402
    autonomous_score,
    build_feature_text,
    labels_to_output,
    prepare_test_features,
    prepare_train_df,
    split_factors,
    train_models,
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
        repeat_assigned=int(params["features"].get("repeat_assigned", 3)),
        include_month=bool(params["features"].get("include_month", True)),
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


def grouped_split(df: pd.DataFrame, params: dict):
    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=float(params["split"]["train_size"]),
        random_state=int(params["split"]["random_state"]),
    )
    groups = df[params["split"]["group_column"]].astype(str)
    train_idx, val_idx = next(splitter.split(df, df["target_any"], groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def factor_prior(df: pd.DataFrame) -> dict[str, float]:
    label_counts = df["target_factors"].explode().dropna().value_counts()
    assigned_counts = df["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    return {
        factor: float(label_counts.get(factor, 0) / max(assigned_counts.get(factor, 1), 1))
        for factor in sorted(label_counts.index.astype(str))
    }


def retrieval_scores(
    neighbors: list[tuple[np.ndarray, np.ndarray]],
    query_assigned: list[list[str]],
    positive_assigned: list[set[str]],
    positive_labels: list[list[str]],
    *,
    same_factor_bonus: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for (idxs, sims), assigned in zip(neighbors, query_assigned):
        assigned_set = set(assigned)
        scores: dict[str, float] = defaultdict(float)
        for neighbor_idx, sim in zip(idxs, sims):
            shared = assigned_set & positive_assigned[neighbor_idx]
            if not shared:
                continue
            weight = float(sim) * (1.0 + same_factor_bonus * len(shared))
            for factor in positive_labels[neighbor_idx]:
                if factor in assigned_set:
                    scores[factor] += weight
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {factor: score / max_score for factor, score in scores.items()}
        rows.append(dict(scores))
    return rows


def hybrid_predict(
    any_scores: np.ndarray,
    factor_proba: np.ndarray,
    retrieval_rows: list[dict[str, float]],
    assigned_lists: list[list[str]],
    classes: np.ndarray,
    priors: dict[str, float],
    *,
    any_threshold: float,
    factor_threshold: float,
    max_factors: int,
    model_weight: float,
    retrieval_weight: float,
    prior_weight: float,
    fallback_top_factor: bool,
) -> list[list[str]]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    predictions: list[list[str]] = []
    for any_score, row_proba, ret_scores, assigned in zip(any_scores, factor_proba, retrieval_rows, assigned_lists):
        allowed = [factor for factor in assigned if factor in class_to_idx]
        if any_score < any_threshold or not allowed:
            predictions.append([])
            continue

        ranked = []
        for factor in allowed:
            score = (
                model_weight * float(row_proba[class_to_idx[factor]])
                + retrieval_weight * float(ret_scores.get(factor, 0.0))
                + prior_weight * float(priors.get(factor, 0.0))
            )
            ranked.append((factor, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = [factor for factor, score in ranked if score >= factor_threshold][:max_factors]
        if not selected and fallback_top_factor:
            selected = [ranked[0][0]]
        predictions.append(list(dict.fromkeys(selected)))
    return predictions


def objective(metrics_by_alpha: dict[str, dict], positive_rate: float, params: dict) -> float:
    base = float(np.mean([m["autonomous_score"] for m in metrics_by_alpha.values()]))
    excess = max(0.0, positive_rate - float(params["prediction"]["positive_rate_soft_max"]))
    return base - float(params["prediction"]["positive_rate_penalty"]) * excess


def update_info(metrics: dict, submission_path: Path) -> None:
    info_path = EXPERIMENT_DIR / "info.yml"
    info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    report = metrics["best_metrics_by_alpha"]["0.5"]
    info["status"] = "done"
    info["local_validation"] = {
        "autonomous_score_alpha_0_5": report["autonomous_score"],
        "objective": metrics["best_objective"],
        "any_f1": report["any_f1"],
        "jaccard": report["factor_jaccard_on_true_positive_rows"],
        "positive_rate_pred": report["positive_rate_pred"],
        "positive_rate_true": report["positive_rate_true"],
    }
    info["outputs"] = {
        "submission": str(submission_path.relative_to(ROOT)),
        "validation_predictions": "experiments/004_hybrid_grouped_ranker/validation_predictions.csv",
        "metrics": "experiments/004_hybrid_grouped_ranker/metrics.json",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    train_df = prepare_train_df(ROOT / params["data"]["train"])
    test_df = pd.read_csv(ROOT / params["data"]["test"])
    train_part, val_part = grouped_split(train_df, params)

    vectorizer, any_model, factor_model, mlb, priors = train_models(train_part, params)
    x_val = vectorizer.transform(prepare_test_features(val_part, params))
    any_scores = any_model.predict_proba(x_val)[:, 1]
    factor_scores = factor_model.predict_proba(x_val)
    assigned_val = val_part["assigned_harmful_factors"].apply(split_factors).tolist()
    true_labels = val_part["target_factors"].tolist()

    pos_part = train_part[train_part["target_any"]].reset_index(drop=True)
    retrieval_vectorizer = make_vectorizer(params)
    x_pos = retrieval_vectorizer.fit_transform(feature_series(pos_part, params))
    x_val_ret = retrieval_vectorizer.transform(feature_series(val_part, params))
    neighbors = top_neighbors(x_val_ret, x_pos, int(params["retrieval"]["top_k"]))
    ret_rows = retrieval_scores(
        neighbors,
        assigned_val,
        pos_part["assigned_harmful_factors"].apply(lambda x: set(split_factors(x))).tolist(),
        pos_part["target_factors"].tolist(),
        same_factor_bonus=float(params["retrieval"]["same_factor_bonus"]),
    )

    best = None
    tried = 0
    grid = params["prediction"]
    for any_threshold in threshold_values(grid["any_threshold_grid"]):
        for factor_threshold in threshold_values(grid["factor_threshold_grid"]):
            for max_factors in grid["max_factors_grid"]:
                for model_weight, retrieval_weight, prior_weight in grid["weight_combinations"]:
                    for fallback_top_factor in grid["fallback_top_factor_grid"]:
                        pred = hybrid_predict(
                            any_scores,
                            factor_scores,
                            ret_rows,
                            assigned_val,
                            mlb.classes_,
                            priors,
                            any_threshold=any_threshold,
                            factor_threshold=factor_threshold,
                            max_factors=int(max_factors),
                            model_weight=float(model_weight),
                            retrieval_weight=float(retrieval_weight),
                            prior_weight=float(prior_weight),
                            fallback_top_factor=bool(fallback_top_factor),
                        )
                        metrics_by_alpha = {
                            str(alpha): autonomous_score(true_labels, pred, alpha=float(alpha))
                            for alpha in params["metric"]["alpha_grid"]
                        }
                        pred_rate = metrics_by_alpha["0.5"]["positive_rate_pred"]
                        obj = objective(metrics_by_alpha, pred_rate, params)
                        tried += 1
                        if best is None or obj > best["objective"]:
                            best = {
                                "objective": round(float(obj), 6),
                                "params": {
                                    "any_threshold": any_threshold,
                                    "factor_threshold": factor_threshold,
                                    "max_factors": int(max_factors),
                                    "model_weight": float(model_weight),
                                    "retrieval_weight": float(retrieval_weight),
                                    "prior_weight": float(prior_weight),
                                    "fallback_top_factor": bool(fallback_top_factor),
                                },
                                "metrics_by_alpha": metrics_by_alpha,
                                "pred": pred,
                            }
                            print("new_best", json.dumps({k: v for k, v in best.items() if k != "pred"}, ensure_ascii=False), flush=True)

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    vectorizer, any_model, factor_model, mlb, priors = train_models(train_df, params)
    x_test = vectorizer.transform(prepare_test_features(test_df, params))
    any_test = any_model.predict_proba(x_test)[:, 1]
    factor_test = factor_model.predict_proba(x_test)

    pos_all = train_df[train_df["target_any"]].reset_index(drop=True)
    retrieval_vectorizer = make_vectorizer(params)
    x_pos_all = retrieval_vectorizer.fit_transform(feature_series(pos_all, params))
    x_test_ret = retrieval_vectorizer.transform(feature_series(test_df, params))
    test_neighbors = top_neighbors(x_test_ret, x_pos_all, int(params["retrieval"]["top_k"]))
    assigned_test = test_df["assigned_harmful_factors"].apply(split_factors).tolist()
    ret_test = retrieval_scores(
        test_neighbors,
        assigned_test,
        pos_all["assigned_harmful_factors"].apply(lambda x: set(split_factors(x))).tolist(),
        pos_all["target_factors"].tolist(),
        same_factor_bonus=float(params["retrieval"]["same_factor_bonus"]),
    )
    test_pred = hybrid_predict(
        any_test,
        factor_test,
        ret_test,
        assigned_test,
        mlb.classes_,
        priors,
        **best["params"],
    )
    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], test_pred)

    metrics = {
        "experiment": params["name"],
        "rows_train_total": int(len(train_df)),
        "rows_train_fit": int(len(train_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "group_split_unique_patients_train": int(train_part["patient_id"].nunique()),
        "group_split_unique_patients_validation": int(val_part["patient_id"].nunique()),
        "target_positive_rate_total": round(float(train_df["target_any"].mean()), 6),
        "target_positive_rate_validation": round(float(val_part["target_any"].mean()), 6),
        "best_objective": best["objective"],
        "best_params": best["params"],
        "best_metrics_by_alpha": best["metrics_by_alpha"],
        "grid_combinations_tried": tried,
        "submission_positive_rows": int(sum(bool(x) for x in test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in test_pred])), 6),
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_info(metrics, submission_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
