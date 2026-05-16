import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import autonomous_score, labels_to_output, prepare_train_df, write_submission  # noqa: E402


def load_exp006():
    path = ROOT / "experiments" / "006_candidate_factor_binary" / "train.py"
    spec = importlib.util.spec_from_file_location("exp006_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def range_values(spec: list[float]) -> list[float]:
    start, stop, step = spec
    return [round(float(x), 4) for x in np.arange(start, stop + step / 2, step)]


def predictions_from_thresholds(
    candidate_df: pd.DataFrame,
    scores: np.ndarray,
    thresholds: dict[str, float],
    default_threshold: float,
    row_order: pd.DataFrame,
) -> list[list[str]]:
    selected_by_row: dict[object, list[tuple[str, float]]] = {}
    factors = candidate_df["factor"].tolist()
    source_indices = candidate_df["source_index"].tolist()
    for factor, row_idx, score in zip(factors, source_indices, scores):
        threshold = thresholds.get(factor, default_threshold)
        if score >= threshold:
            selected_by_row.setdefault(row_idx, []).append((factor, float(score)))

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
    target_penalty = 0.35 * abs(pred_rate - float(cfg["target_positive_rate"]))
    return (
        metrics["autonomous_score"]
        + float(cfg["jaccard_weight_bonus"]) * metrics["factor_jaccard_on_true_positive_rows"]
        - penalty
        - target_penalty
    )


def evaluate(candidate_df, scores, thresholds, default_threshold, val_part, params):
    pred = predictions_from_thresholds(candidate_df, scores, thresholds, default_threshold, val_part)
    metrics = autonomous_score(val_part["target_factors"].tolist(), pred, alpha=float(params["metric"]["alpha"]))
    return objective(metrics, params), metrics, pred


def optimize_thresholds(candidate_df, scores, val_part, params):
    best = None
    for threshold in range_values(params["prediction"]["global_threshold_grid"]):
        obj, metrics, pred = evaluate(candidate_df, scores, {}, threshold, val_part, params)
        if best is None or obj > best["objective"]:
            best = {
                "objective": round(float(obj), 6),
                "default_threshold": threshold,
                "thresholds": {},
                "metrics": metrics,
                "pred": pred,
            }

    factor_counts = candidate_df["factor"].value_counts()
    factors = factor_counts[factor_counts >= int(params["prediction"]["per_factor_min_candidates"])].index.tolist()
    grid = range_values(params["prediction"]["per_factor_threshold_grid"])

    for _ in range(int(params["prediction"]["coordinate_passes"])):
        improved = False
        for factor in factors:
            current_best = best
            for threshold in grid:
                thresholds = dict(best["thresholds"])
                thresholds[factor] = threshold
                obj, metrics, pred = evaluate(candidate_df, scores, thresholds, best["default_threshold"], val_part, params)
                if obj > current_best["objective"]:
                    current_best = {
                        "objective": round(float(obj), 6),
                        "default_threshold": best["default_threshold"],
                        "thresholds": thresholds,
                        "metrics": metrics,
                        "pred": pred,
                    }
            if current_best is not best:
                best = current_best
                improved = True
                print("new_best", json.dumps({
                    "objective": best["objective"],
                    "default_threshold": best["default_threshold"],
                    "factor": factor,
                    "factor_threshold": best["thresholds"].get(factor),
                    "metrics": best["metrics"],
                    "n_factor_thresholds": len(best["thresholds"]),
                }, ensure_ascii=False), flush=True)
        if not improved:
            break
    return best


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
        "validation_predictions": "experiments/008_candidate_factor_thresholds/validation_predictions.csv",
        "metrics": "experiments/008_candidate_factor_thresholds/metrics.json",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    exp006 = load_exp006()
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    train_df = prepare_train_df(ROOT / params["data"]["train"])
    test_df = pd.read_csv(ROOT / params["data"]["test"])

    train_part, val_part = train_test_split(
        train_df,
        train_size=float(params["split"]["train_size"]),
        random_state=int(params["split"]["random_state"]),
        stratify=train_df["target_any"],
    )

    train_candidates = exp006.make_candidate_frame(train_part, params, has_target=True)
    val_candidates = exp006.make_candidate_frame(val_part, params, has_target=True)

    vectorizer = exp006.DualTfidf(params)
    x_train = vectorizer.fit_transform(train_candidates["text"].tolist())
    model = LogisticRegression(**params["model"])
    model.fit(x_train, train_candidates["target"].astype(int))

    x_val = vectorizer.transform(val_candidates["text"].tolist())
    val_scores = model.predict_proba(x_val)[:, 1]
    best = optimize_thresholds(val_candidates, val_scores, val_part, params)

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    all_candidates = exp006.make_candidate_frame(train_df, params, has_target=True)
    vectorizer = exp006.DualTfidf(params)
    x_all = vectorizer.fit_transform(all_candidates["text"].tolist())
    model = LogisticRegression(**params["model"])
    model.fit(x_all, all_candidates["target"].astype(int))

    test_candidates = exp006.make_candidate_frame(test_df, params, has_target=False)
    x_test = vectorizer.transform(test_candidates["text"].tolist())
    test_scores = model.predict_proba(x_test)[:, 1]
    test_pred = predictions_from_thresholds(
        test_candidates,
        test_scores,
        best["thresholds"],
        best["default_threshold"],
        test_df,
    )

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
        "default_threshold": best["default_threshold"],
        "n_factor_thresholds": len(best["thresholds"]),
        "factor_thresholds": best["thresholds"],
        "best_metrics": best["metrics"],
        "submission_positive_rows": int(sum(bool(x) for x in test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in test_pred])), 6),
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_info(metrics, submission_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
