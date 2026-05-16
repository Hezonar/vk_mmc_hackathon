import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "reports" / "ml_baseline"


def split_factors(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def clean_factor_token(code: str) -> str:
    return "factor_" + re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", code).strip("_")


def parse_specialists(value: object) -> list[dict]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_feature_text(row: pd.Series) -> str:
    parts: list[str] = []

    assigned = split_factors(row.get("assigned_harmful_factors", ""))
    for factor in assigned:
        token = clean_factor_token(factor)
        parts.extend([token, token])

    date = str(row.get("consultation_date", ""))
    if len(date) >= 7:
        parts.append("month_" + date[:7])

    for item in parse_specialists(row.get("specialist_conclusions", "")):
        specialist = str(item.get("specialist", "")).strip()
        health_group = str(item.get("health_group", "")).strip()
        mkb_code = str(item.get("mkb_code", "")).strip()
        mkb_description = str(item.get("mkb_description", "")).strip()
        conclusion = str(item.get("conclusion", "")).strip()

        if specialist:
            parts.append("specialist_" + specialist)
        if health_group:
            parts.append("health_" + health_group)
        if mkb_code:
            parts.extend(["mkb_" + mkb_code, "mkb_prefix_" + mkb_code[:1]])
        if mkb_description:
            parts.append(mkb_description)
        if conclusion:
            parts.append(conclusion)

    return " ".join(parts).lower()


def labels_to_output(labels: list[str]) -> str:
    return ";".join(labels) if labels else "0"


def sample_f1(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    scores = []
    for true_labels, pred_labels in zip(y_true, y_pred):
        true_set = set(true_labels)
        pred_set = set(pred_labels)
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set or not pred_set:
            scores.append(0.0)
        else:
            scores.append(2 * len(true_set & pred_set) / (len(true_set) + len(pred_set)))
    return float(np.mean(scores))


def predict_factor_lists(
    any_proba: np.ndarray,
    factor_proba: np.ndarray,
    assigned_lists: list[list[str]],
    classes: np.ndarray,
    factor_positive_rate: dict[str, float],
    any_threshold: float,
    factor_threshold: float,
) -> list[list[str]]:
    predictions: list[list[str]] = []
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    for p_any, row_factor_proba, assigned in zip(any_proba, factor_proba, assigned_lists):
        if p_any < any_threshold:
            predictions.append([])
            continue

        allowed = [factor for factor in assigned if factor in class_to_idx]
        selected = [
            factor for factor in allowed
            if row_factor_proba[class_to_idx[factor]] >= factor_threshold
        ]

        if not selected and allowed:
            selected = [max(allowed, key=lambda factor: (
                row_factor_proba[class_to_idx[factor]],
                factor_positive_rate.get(factor, 0.0),
            ))]

        predictions.append(list(dict.fromkeys(selected)))

    return predictions


def evaluate(true_labels: list[list[str]], pred_labels: list[list[str]], mlb_eval: MultiLabelBinarizer) -> dict:
    y_true_any = np.array([bool(x) for x in true_labels])
    y_pred_any = np.array([bool(x) for x in pred_labels])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_any, y_pred_any, average="binary", zero_division=0
    )

    y_true_multi = mlb_eval.transform(true_labels)
    y_pred_multi = mlb_eval.transform(pred_labels)
    return {
        "any_precision": round(float(precision), 4),
        "any_recall": round(float(recall), 4),
        "any_f1": round(float(f1), 4),
        "factor_micro_f1": round(float(f1_score(y_true_multi, y_pred_multi, average="micro", zero_division=0)), 4),
        "factor_samples_f1": round(sample_f1(true_labels, pred_labels), 4),
        "positive_rate_pred": round(float(y_pred_any.mean()), 4),
        "positive_rate_true": round(float(y_true_any.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(DATA_DIR / "train2.csv"))
    parser.add_argument("--test", default=str(DATA_DIR / "test2.csv"))
    parser.add_argument("--submission", default=str(OUT_DIR / "submission_baseline.csv"))
    parser.add_argument("--validation-predictions", default=str(OUT_DIR / "validation_predictions.csv"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    train_df["target_factors"] = train_df["contraindicated_factors"].apply(split_factors)
    train_df["target_any"] = train_df["target_factors"].apply(bool)
    train_df["feature_text"] = train_df.apply(build_feature_text, axis=1)
    test_df["feature_text"] = test_df.apply(build_feature_text, axis=1)
    mlb_eval = MultiLabelBinarizer().fit(train_df["target_factors"])

    train_part, val_part = train_test_split(
        train_df,
        train_size=0.70,
        random_state=args.random_state,
        stratify=train_df["target_any"],
    )

    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=120_000,
    )
    x_train = vectorizer.fit_transform(train_part["feature_text"])
    x_val = vectorizer.transform(val_part["feature_text"])

    any_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=args.random_state,
    )
    any_model.fit(x_train, train_part["target_any"].astype(int))

    mlb = MultiLabelBinarizer()
    y_train_multi = mlb.fit_transform(train_part["target_factors"])
    factor_model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=args.random_state,
        )
    )
    factor_model.fit(x_train, y_train_multi)

    train_label_counts = train_part["target_factors"].explode().dropna().value_counts()
    train_assigned_counts = train_part["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    factor_positive_rate = {
        factor: float(train_label_counts.get(factor, 0) / max(train_assigned_counts.get(factor, 1), 1))
        for factor in mlb.classes_
    }

    any_val_proba = any_model.predict_proba(x_val)[:, 1]
    factor_val_proba = factor_model.predict_proba(x_val)
    val_assigned = val_part["assigned_harmful_factors"].apply(split_factors).tolist()
    true_val_labels = val_part["target_factors"].tolist()

    best = None
    for any_threshold in np.arange(0.15, 0.71, 0.05):
        for factor_threshold in np.arange(0.15, 0.71, 0.05):
            pred = predict_factor_lists(
                any_val_proba,
                factor_val_proba,
                val_assigned,
                mlb.classes_,
                factor_positive_rate,
                float(any_threshold),
                float(factor_threshold),
            )
            metrics = evaluate(true_val_labels, pred, mlb_eval)
            score = metrics["factor_micro_f1"] + 0.15 * metrics["any_f1"]
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "any_threshold": round(float(any_threshold), 2),
                    "factor_threshold": round(float(factor_threshold), 2),
                    "metrics": metrics,
                    "pred": pred,
                }

    val_out = val_part[["exam_row_id", "contraindicated_factors"]].copy()
    val_out["predicted_factors"] = [labels_to_output(x) for x in best["pred"]]
    val_out.to_csv(args.validation_predictions, index=False)

    x_all = vectorizer.fit_transform(train_df["feature_text"])
    any_model.fit(x_all, train_df["target_any"].astype(int))
    mlb = MultiLabelBinarizer()
    y_all_multi = mlb.fit_transform(train_df["target_factors"])
    factor_model.fit(x_all, y_all_multi)

    all_label_counts = train_df["target_factors"].explode().dropna().value_counts()
    all_assigned_counts = train_df["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    factor_positive_rate = {
        factor: float(all_label_counts.get(factor, 0) / max(all_assigned_counts.get(factor, 1), 1))
        for factor in mlb.classes_
    }

    x_test = vectorizer.transform(test_df["feature_text"])
    test_pred = predict_factor_lists(
        any_model.predict_proba(x_test)[:, 1],
        factor_model.predict_proba(x_test),
        test_df["assigned_harmful_factors"].apply(split_factors).tolist(),
        mlb.classes_,
        factor_positive_rate,
        best["any_threshold"],
        best["factor_threshold"],
    )

    submission = pd.DataFrame({
        "exam_row_id": test_df["exam_row_id"],
        "factors": [labels_to_output(x) for x in test_pred],
    })
    submission.to_csv(args.submission, index=False)

    summary = {
        "train_path": str(Path(args.train).resolve()),
        "test_path": str(Path(args.test).resolve()),
        "rows_train_total": int(len(train_df)),
        "rows_train_fit": int(len(train_part)),
        "rows_validation": int(len(val_part)),
        "rows_test": int(len(test_df)),
        "target_positive_rate": round(float(train_df["target_any"].mean()), 4),
        "n_factor_classes": int(len(mlb.classes_)),
        "best_thresholds": {
            "any_threshold": best["any_threshold"],
            "factor_threshold": best["factor_threshold"],
        },
        "validation_metrics": best["metrics"],
        "submission_positive_rows": int((submission["factors"] != "0").sum()),
        "submission_positive_rate": round(float((submission["factors"] != "0").mean()), 4),
        "submission_path": str(Path(args.submission).resolve()),
        "validation_predictions_path": str(Path(args.validation_predictions).resolve()),
    }
    (OUT_DIR / "baseline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
