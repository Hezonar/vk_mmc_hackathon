import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import (  # noqa: E402
    autonomous_score,
    build_feature_text,
    labels_to_output,
    predict_factor_lists,
    prepare_test_features,
    prepare_train_df,
    split_factors,
    split_train_valid,
    train_models,
    write_submission,
)


def threshold_values(spec: list[float]) -> list[float]:
    start, stop, step = spec
    return [round(float(x), 4) for x in np.arange(start, stop + step / 2, step)]


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
        "validation_predictions": "experiments/002_autonomous_tuned/validation_predictions.csv",
        "metrics": "experiments/002_autonomous_tuned/metrics.json",
    }
    info_path.write_text(yaml.safe_dump(info, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    params = yaml.safe_load((EXPERIMENT_DIR / "params.yml").read_text(encoding="utf-8"))
    alpha = float(params["metric"]["alpha"])

    train_path = ROOT / params["data"]["train"]
    test_path = ROOT / params["data"]["test"]
    train_df = prepare_train_df(train_path)
    test_df = pd.read_csv(test_path)

    train_part, val_part = split_train_valid(train_df, params)
    vectorizer, any_model, factor_model, mlb, factor_prior = train_models(train_part, params)

    val_features = val_part.apply(
        lambda row: build_feature_text(
            row,
            repeat_assigned=int(params["features"].get("repeat_assigned", 2)),
            include_month=bool(params["features"].get("include_month", True)),
        ),
        axis=1,
    )
    x_val = vectorizer.transform(val_features)
    any_scores = any_model.predict_proba(x_val)[:, 1]
    factor_scores = factor_model.predict_proba(x_val)
    assigned_lists = val_part["assigned_harmful_factors"].apply(split_factors).tolist()
    true_labels = val_part["target_factors"].tolist()

    best = None
    pred_grid = params["prediction"]
    for any_threshold in threshold_values(pred_grid["any_threshold_grid"]):
        for factor_threshold in threshold_values(pred_grid["factor_threshold_grid"]):
            for max_factors in pred_grid["max_factors_grid"]:
                for fallback_top_factor in pred_grid["fallback_top_factor_grid"]:
                    pred = predict_factor_lists(
                        any_scores,
                        factor_scores,
                        assigned_lists,
                        mlb.classes_,
                        factor_prior,
                        any_threshold=any_threshold,
                        factor_threshold=factor_threshold,
                        max_factors=int(max_factors),
                        fallback_top_factor=bool(fallback_top_factor),
                    )
                    metrics = autonomous_score(true_labels, pred, alpha=alpha)
                    if best is None or metrics["autonomous_score"] > best["metrics"]["autonomous_score"]:
                        best = {
                            "any_threshold": any_threshold,
                            "factor_threshold": factor_threshold,
                            "max_factors": int(max_factors),
                            "fallback_top_factor": bool(fallback_top_factor),
                            "metrics": metrics,
                            "pred": pred,
                        }

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    vectorizer, any_model, factor_model, mlb, factor_prior = train_models(train_df, params)
    test_features = prepare_test_features(test_df, params)
    x_test = vectorizer.transform(test_features)
    test_pred = predict_factor_lists(
        any_model.predict_proba(x_test)[:, 1],
        factor_model.predict_proba(x_test),
        test_df["assigned_harmful_factors"].apply(split_factors).tolist(),
        mlb.classes_,
        factor_prior,
        any_threshold=best["any_threshold"],
        factor_threshold=best["factor_threshold"],
        max_factors=best["max_factors"],
        fallback_top_factor=best["fallback_top_factor"],
    )

    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], test_pred)

    metrics = {
        "experiment": params["name"],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "rows_train_total": int(len(train_df)),
        "rows_train_fit": int(len(train_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "n_factor_classes": int(len(mlb.classes_)),
        "target_positive_rate": round(float(train_df["target_any"].mean()), 6),
        "best_params": {
            "any_threshold": best["any_threshold"],
            "factor_threshold": best["factor_threshold"],
            "max_factors": best["max_factors"],
            "fallback_top_factor": best["fallback_top_factor"],
        },
        "best_metrics": best["metrics"],
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
