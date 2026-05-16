import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def split_factors(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "0":
        return []
    return [x.strip() for x in text.split(";") if x.strip()]


def labels_to_output(labels: list[str]) -> str:
    return ";".join(labels) if labels else "0"


def clean_token(value: str, prefix: str) -> str:
    token = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", value).strip("_")
    return f"{prefix}_{token}" if token else prefix


def parse_specialists(value: object) -> list[dict[str, Any]]:
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


def build_feature_text(row: pd.Series, *, repeat_assigned: int = 2, include_month: bool = True) -> str:
    parts: list[str] = []

    for factor in split_factors(row.get("assigned_harmful_factors", "")):
        parts.extend([clean_token(factor, "factor")] * repeat_assigned)

    date = str(row.get("consultation_date", ""))
    if include_month and len(date) >= 7:
        parts.append("month_" + date[:7])

    for item in parse_specialists(row.get("specialist_conclusions", "")):
        specialist = str(item.get("specialist", "")).strip()
        health_group = str(item.get("health_group", "")).strip()
        mkb_code = str(item.get("mkb_code", "")).strip()
        mkb_description = str(item.get("mkb_description", "")).strip()
        conclusion = str(item.get("conclusion", "")).strip()

        if specialist:
            parts.append(clean_token(specialist, "specialist"))
        if health_group:
            parts.append(clean_token(health_group, "health"))
        if mkb_code:
            parts.extend([clean_token(mkb_code, "mkb"), "mkb_prefix_" + mkb_code[:1].lower()])
        if mkb_description:
            parts.append(mkb_description)
        if conclusion:
            parts.append(conclusion)

    return " ".join(parts).lower()


def autonomous_score(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    true_any = np.array([bool(x) for x in y_true])
    pred_any = np.array([bool(x) for x in y_pred])
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_any, pred_any, average="binary", zero_division=0
    )

    jaccards = []
    for true_labels, pred_labels in zip(y_true, y_pred):
        true_set = set(true_labels)
        if not true_set:
            continue
        pred_set = set(pred_labels)
        union = true_set | pred_set
        jaccards.append(len(true_set & pred_set) / len(union) if union else 0.0)

    jaccard = float(np.mean(jaccards)) if jaccards else 0.0
    score = alpha * float(f1) + (1.0 - alpha) * jaccard
    return {
        "autonomous_score": round(score, 6),
        "alpha": alpha,
        "any_precision": round(float(precision), 6),
        "any_recall": round(float(recall), 6),
        "any_f1": round(float(f1), 6),
        "factor_jaccard_on_true_positive_rows": round(jaccard, 6),
        "positive_rate_true": round(float(true_any.mean()), 6),
        "positive_rate_pred": round(float(pred_any.mean()), 6),
    }


def predict_factor_lists(
    any_scores: np.ndarray,
    factor_scores: np.ndarray,
    assigned_lists: list[list[str]],
    classes: np.ndarray,
    factor_prior: dict[str, float],
    *,
    any_threshold: float,
    factor_threshold: float,
    max_factors: int = 3,
    fallback_top_factor: bool = True,
) -> list[list[str]]:
    predictions: list[list[str]] = []
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    for any_score, row_scores, assigned in zip(any_scores, factor_scores, assigned_lists):
        allowed = [factor for factor in assigned if factor in class_to_idx]
        if any_score < any_threshold or not allowed:
            predictions.append([])
            continue

        ranked = sorted(
            allowed,
            key=lambda factor: (
                row_scores[class_to_idx[factor]],
                factor_prior.get(factor, 0.0),
            ),
            reverse=True,
        )
        selected = [
            factor for factor in ranked
            if row_scores[class_to_idx[factor]] >= factor_threshold
        ][:max_factors]

        if not selected and fallback_top_factor:
            selected = ranked[:1]

        predictions.append(list(dict.fromkeys(selected)))

    return predictions


def train_models(train_df: pd.DataFrame, params: dict[str, Any]):
    train_df = train_df.copy()
    train_df["feature_text"] = train_df.apply(
        build_feature_text,
        axis=1,
        repeat_assigned=int(params["features"].get("repeat_assigned", 2)),
        include_month=bool(params["features"].get("include_month", True)),
    )

    vectorizer_params = dict(params["vectorizer"])
    if "ngram_range" in vectorizer_params:
        vectorizer_params["ngram_range"] = tuple(vectorizer_params["ngram_range"])
    vectorizer = TfidfVectorizer(**vectorizer_params)
    x_train = vectorizer.fit_transform(train_df["feature_text"])

    any_model = LogisticRegression(**params["any_model"])
    any_model.fit(x_train, train_df["target_any"].astype(int))

    mlb = MultiLabelBinarizer()
    y_multi = mlb.fit_transform(train_df["target_factors"])
    factor_model = OneVsRestClassifier(LogisticRegression(**params["factor_model"]))
    factor_model.fit(x_train, y_multi)

    label_counts = train_df["target_factors"].explode().dropna().value_counts()
    assigned_counts = train_df["assigned_harmful_factors"].apply(split_factors).explode().dropna().value_counts()
    factor_prior = {
        factor: float(label_counts.get(factor, 0) / max(assigned_counts.get(factor, 1), 1))
        for factor in mlb.classes_
    }
    return vectorizer, any_model, factor_model, mlb, factor_prior


def prepare_train_df(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target_factors"] = df["contraindicated_factors"].apply(split_factors)
    df["target_any"] = df["target_factors"].apply(bool)
    return df


def prepare_test_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    return df.apply(
        build_feature_text,
        axis=1,
        repeat_assigned=int(params["features"].get("repeat_assigned", 2)),
        include_month=bool(params["features"].get("include_month", True)),
    )


def split_train_valid(df: pd.DataFrame, params: dict[str, Any]):
    return train_test_split(
        df,
        train_size=float(params["split"]["train_size"]),
        random_state=int(params["split"]["random_state"]),
        stratify=df["target_any"],
    )


def write_submission(path: str | Path, exam_row_id: pd.Series, predictions: list[list[str]]) -> None:
    out = pd.DataFrame({
        "exam_row_id": exam_row_id,
        "factors": [labels_to_output(x) for x in predictions],
    })
    out.to_csv(path, index=False)
