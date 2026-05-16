import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

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


def make_features(df: pd.DataFrame, params: dict) -> pd.Series:
    return df.apply(
        build_feature_text,
        axis=1,
        repeat_assigned=int(params["features"]["repeat_assigned"]),
        include_month=bool(params["features"]["include_month"]),
    )


class DualTfidf:
    def __init__(self, params: dict):
        self.word = TfidfVectorizer(**vectorizer_params(params, "word_vectorizer"))
        self.char = TfidfVectorizer(**vectorizer_params(params, "char_vectorizer"))

    def fit_transform(self, texts: pd.Series):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)], format="csr")

    def transform(self, texts: pd.Series):
        return hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")


def factor_prior(df: pd.DataFrame, classes: np.ndarray) -> dict[str, float]:
    label_counts = df["target_factors"].explode().dropna().value_counts()
    assigned_counts = df["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    return {
        factor: float(label_counts.get(factor, 0) / max(assigned_counts.get(factor, 1), 1))
        for factor in classes
    }


def objective(metrics: dict, params: dict) -> float:
    pred_rate = metrics["positive_rate_pred"]
    cfg = params["prediction"]
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


def train_core(train_df: pd.DataFrame, params: dict):
    texts = make_features(train_df, params)
    vectorizer = DualTfidf(params)
    x_train = vectorizer.fit_transform(texts)

    any_model = LogisticRegression(**params["any_model"])
    any_model.fit(x_train, train_df["target_any"].astype(int))

    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(train_df["target_factors"])
    factor_model = OneVsRestClassifier(LogisticRegression(**params["factor_model"]))
    factor_model.fit(x_train, y_multi)
    return vectorizer, any_model, factor_model, mlb, factor_prior(train_df, mlb.classes_)


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
        "validation_predictions": "experiments/005_conservative_tfidf/validation_predictions.csv",
        "metrics": "experiments/005_conservative_tfidf/metrics.json",
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

    vectorizer, any_model, factor_model, mlb, priors = train_core(train_part, params)
    x_val = vectorizer.transform(make_features(val_part, params))
    any_scores = any_model.predict_proba(x_val)[:, 1]
    factor_scores = factor_model.predict_proba(x_val)
    assigned_val = val_part["assigned_harmful_factors"].apply(split_factors).tolist()
    true_labels = val_part["target_factors"].tolist()

    best = None
    tried = 0
    grid = params["prediction"]
    for any_threshold in range_values(grid["any_threshold_grid"]):
        for factor_threshold in range_values(grid["factor_threshold_grid"]):
            for max_factors in grid["max_factors_grid"]:
                for fallback_top_factor in grid["fallback_top_factor_grid"]:
                    pred = predict_factor_lists(
                        any_scores,
                        factor_scores,
                        assigned_val,
                        mlb.classes_,
                        priors,
                        any_threshold=any_threshold,
                        factor_threshold=factor_threshold,
                        max_factors=int(max_factors),
                        fallback_top_factor=bool(fallback_top_factor),
                    )
                    metrics = autonomous_score(true_labels, pred, alpha=float(params["metric"]["alpha"]))
                    obj = objective(metrics, params)
                    tried += 1
                    if best is None or obj > best["objective"]:
                        best = {
                            "objective": round(float(obj), 6),
                            "params": {
                                "any_threshold": any_threshold,
                                "factor_threshold": factor_threshold,
                                "max_factors": int(max_factors),
                                "fallback_top_factor": bool(fallback_top_factor),
                            },
                            "metrics": metrics,
                            "pred": pred,
                        }
                        print("new_best", json.dumps({k: v for k, v in best.items() if k != "pred"}, ensure_ascii=False), flush=True)

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    vectorizer, any_model, factor_model, mlb, priors = train_core(train_df, params)
    x_test = vectorizer.transform(prepare_test_features(test_df, params))
    test_pred = predict_factor_lists(
        any_model.predict_proba(x_test)[:, 1],
        factor_model.predict_proba(x_test),
        test_df["assigned_harmful_factors"].apply(split_factors).tolist(),
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
