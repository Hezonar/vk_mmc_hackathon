import importlib.util
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

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.append(str(ROOT / "experiments" / "common"))

from medical_factor_baseline import autonomous_score, build_feature_text, labels_to_output, prepare_train_df, split_factors, write_submission  # noqa: E402


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


def vectorizer_params(params: dict, key: str) -> dict:
    out = dict(params[key])
    if "ngram_range" in out:
        out["ngram_range"] = tuple(out["ngram_range"])
    return out


class DualVectorizer:
    def __init__(self, params: dict, word_key: str, char_key: str):
        self.word = TfidfVectorizer(**vectorizer_params(params, word_key))
        self.char = TfidfVectorizer(**vectorizer_params(params, char_key))

    def fit(self, texts: list[str]) -> None:
        self.word.fit(texts)
        self.char.fit(texts)

    def transform(self, texts: list[str]):
        return hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")


def row_texts(df: pd.DataFrame, params: dict) -> list[str]:
    return [
        build_feature_text(
            row,
            repeat_assigned=int(params["row_features"]["repeat_assigned"]),
            include_month=bool(params["row_features"]["include_month"]),
        )
        for _, row in df.iterrows()
    ]


def rows_from_scores(
    candidate_df: pd.DataFrame,
    candidate_scores: np.ndarray,
    row_scores: pd.Series,
    *,
    row_threshold: float,
    factor_threshold: float,
    max_factors: int,
    row_order: pd.DataFrame,
) -> list[list[str]]:
    selected_by_row: dict[object, list[tuple[str, float]]] = {}
    for (_, row), score in zip(candidate_df.iterrows(), candidate_scores):
        source_index = row["source_index"]
        if float(row_scores.loc[source_index]) < row_threshold:
            continue
        if score >= factor_threshold:
            selected_by_row.setdefault(source_index, []).append((row["factor"], float(score)))

    predictions = []
    for row_idx in row_order.index:
        selected = sorted(selected_by_row.get(row_idx, []), key=lambda x: x[1], reverse=True)[:max_factors]
        predictions.append([factor for factor, _ in selected])
    return predictions


def objective(metrics: dict, test_positive_rate: float, params: dict) -> float:
    cfg = params["prediction"]
    score = metrics["autonomous_score"] + float(cfg["jaccard_weight_bonus"]) * metrics["factor_jaccard_on_true_positive_rows"]

    if test_positive_rate < float(cfg["test_positive_rate_min"]):
        score -= float(cfg["test_positive_rate_penalty"]) * (float(cfg["test_positive_rate_min"]) - test_positive_rate)
    if test_positive_rate > float(cfg["test_positive_rate_max"]):
        score -= float(cfg["test_positive_rate_penalty"]) * (test_positive_rate - float(cfg["test_positive_rate_max"]))
    score -= 0.25 * abs(test_positive_rate - float(cfg["test_positive_rate_target"]))

    val_rate = metrics["positive_rate_pred"]
    if val_rate < float(cfg["validation_positive_rate_soft_min"]):
        score -= float(cfg["validation_positive_rate_penalty"]) * (float(cfg["validation_positive_rate_soft_min"]) - val_rate)
    if val_rate > float(cfg["validation_positive_rate_soft_max"]):
        score -= float(cfg["validation_positive_rate_penalty"]) * (val_rate - float(cfg["validation_positive_rate_soft_max"]))
    return float(score)


def train_models(exp006, train_part: pd.DataFrame, test_df: pd.DataFrame, params: dict):
    row_vec = DualVectorizer(params, "row_word_vectorizer", "row_char_vectorizer")
    row_vec.fit(row_texts(pd.concat([train_part, test_df], axis=0, ignore_index=True), params))
    x_row_train = row_vec.transform(row_texts(train_part, params))
    row_model = LogisticRegression(**params["row_model"])
    row_model.fit(x_row_train, train_part["target_any"].astype(int))

    candidate_train = exp006.make_candidate_frame(train_part, candidate_params(params), has_target=True)
    candidate_test = exp006.make_candidate_frame(test_df, candidate_params(params), has_target=False)
    cand_vec = DualVectorizer(params, "candidate_word_vectorizer", "candidate_char_vectorizer")
    cand_vec.fit(candidate_train["text"].tolist() + candidate_test["text"].tolist())
    x_cand_train = cand_vec.transform(candidate_train["text"].tolist())
    cand_model = LogisticRegression(**params["candidate_model"])
    cand_model.fit(x_cand_train, candidate_train["target"].astype(int))
    return row_vec, row_model, cand_vec, cand_model


def candidate_params(params: dict) -> dict:
    # exp006.make_candidate_frame expects keys named like its own params file.
    out = dict(params)
    out["features"] = dict(params["candidate_features"])
    out["word_vectorizer"] = dict(params["candidate_word_vectorizer"])
    out["char_vectorizer"] = dict(params["candidate_char_vectorizer"])
    out["model"] = dict(params["candidate_model"])
    return out


def score_rows(row_vec, row_model, df: pd.DataFrame, params: dict) -> pd.Series:
    scores = row_model.predict_proba(row_vec.transform(row_texts(df, params)))[:, 1]
    return pd.Series(scores, index=df.index)


def score_candidates(exp006, cand_vec, cand_model, df: pd.DataFrame, params: dict, *, has_target: bool):
    candidates = exp006.make_candidate_frame(df, candidate_params(params), has_target=has_target)
    scores = cand_model.predict_proba(cand_vec.transform(candidates["text"].tolist()))[:, 1] if len(candidates) else np.array([])
    return candidates, scores


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
        "validation_predictions": "experiments/010_test_aware_gate_candidate/validation_predictions.csv",
        "metrics": "experiments/010_test_aware_gate_candidate/metrics.json",
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

    row_vec, row_model, cand_vec, cand_model = train_models(exp006, train_part, test_df, params)
    val_row_scores = score_rows(row_vec, row_model, val_part, params)
    test_row_scores = score_rows(row_vec, row_model, test_df, params)
    val_candidates, val_candidate_scores = score_candidates(exp006, cand_vec, cand_model, val_part, params, has_target=True)
    test_candidates, test_candidate_scores = score_candidates(exp006, cand_vec, cand_model, test_df, params, has_target=False)

    best = None
    tried = 0
    true_labels = val_part["target_factors"].tolist()
    for row_threshold in range_values(params["prediction"]["row_threshold_grid"]):
        for factor_threshold in range_values(params["prediction"]["factor_threshold_grid"]):
            for max_factors in params["prediction"]["max_factors_grid"]:
                val_pred = rows_from_scores(
                    val_candidates,
                    val_candidate_scores,
                    val_row_scores,
                    row_threshold=row_threshold,
                    factor_threshold=factor_threshold,
                    max_factors=int(max_factors),
                    row_order=val_part,
                )
                test_pred = rows_from_scores(
                    test_candidates,
                    test_candidate_scores,
                    test_row_scores,
                    row_threshold=row_threshold,
                    factor_threshold=factor_threshold,
                    max_factors=int(max_factors),
                    row_order=test_df,
                )
                metrics = autonomous_score(true_labels, val_pred, alpha=float(params["metric"]["alpha"]))
                test_positive_rate = float(np.mean([bool(x) for x in test_pred]))
                obj = objective(metrics, test_positive_rate, params)
                tried += 1
                if best is None or obj > best["objective"]:
                    best = {
                        "objective": round(obj, 6),
                        "params": {
                            "row_threshold": row_threshold,
                            "factor_threshold": factor_threshold,
                            "max_factors": int(max_factors),
                        },
                        "metrics": metrics,
                        "test_positive_rate": round(test_positive_rate, 6),
                        "val_pred": val_pred,
                        "test_pred": test_pred,
                    }
                    print("new_best", json.dumps({k: v for k, v in best.items() if k not in {"val_pred", "test_pred"}}, ensure_ascii=False), flush=True)

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["val_pred"]]
    val_out.to_csv(EXPERIMENT_DIR / "validation_predictions.csv", index=False)

    # Refit on all train rows, still fitting vectorizers on train+test texts.
    row_vec, row_model, cand_vec, cand_model = train_models(exp006, train_df, test_df, params)
    all_test_row_scores = score_rows(row_vec, row_model, test_df, params)
    all_test_candidates, all_test_candidate_scores = score_candidates(exp006, cand_vec, cand_model, test_df, params, has_target=False)
    final_test_pred = rows_from_scores(
        all_test_candidates,
        all_test_candidate_scores,
        all_test_row_scores,
        row_threshold=best["params"]["row_threshold"],
        factor_threshold=best["params"]["factor_threshold"],
        max_factors=best["params"]["max_factors"],
        row_order=test_df,
    )

    submission_path = EXPERIMENT_DIR / "submission.csv"
    write_submission(submission_path, test_df["exam_row_id"], final_test_pred)

    metrics = {
        "experiment": params["name"],
        "rows_train_total": int(len(train_df)),
        "rows_train_fit": int(len(train_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "best_objective": best["objective"],
        "best_params": best["params"],
        "best_metrics": best["metrics"],
        "validation_test_positive_rate_during_search": best["test_positive_rate"],
        "grid_combinations_tried": tried,
        "submission_positive_rows": int(sum(bool(x) for x in final_test_pred)),
        "submission_positive_rate": round(float(np.mean([bool(x) for x in final_test_pred])), 6),
    }
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    update_info(metrics, submission_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
