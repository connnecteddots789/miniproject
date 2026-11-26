"""Reproduce the separate UNSW-NB15 binary intrusion-detection benchmark.

This is deliberately separate from the app's native 10-class attack_cat task.
The official testing CSV is used once, after model selection on a validation split
drawn only from the official training CSV.
"""
from __future__ import annotations
import json, time
from pathlib import Path

import joblib
import sys
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app.datasets import load_unsw, feature_frame
from app.ml import make_preprocessor
DATA = ROOT / "backend" / "data" / "unsw_nb15"
OUT = ROOT / "backend" / "artifacts" / "research" / "unsw_nb15_binary_lightgbm"
SEED = 42
CANDIDATES = [
    {"n_estimators": 220, "num_leaves": 31, "learning_rate": 0.06, "min_child_samples": 20},
    {"n_estimators": 350, "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 30},
    {"n_estimators": 260, "num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 30},
    {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.04, "min_child_samples": 40},
]
LABELS = ["Normal", "Attack"]


def main():
    train_path = DATA / "UNSW_NB15_training-set.csv"
    test_path = DATA / "UNSW_NB15_testing-set.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Both official UNSW-NB15 train/test CSV files are required.")

    train_df = load_unsw(train_path)
    test_df = load_unsw(test_path)
    if "label" not in train_df or "label" not in test_df:
        raise ValueError("UNSW-NB15 binary mode requires the official 'label' target column.")

    # The attack_cat and label columns are both excluded from features by feature_frame.
    X = feature_frame(train_df, "UNSW-NB15", "attack_cat")
    X_test = feature_frame(test_df, "UNSW-NB15", "attack_cat").reindex(columns=X.columns)
    y = np.where(pd_to_int(train_df["label"]) == 0, "Normal", "Attack")
    y_test = np.where(pd_to_int(test_df["label"]) == 0, "Normal", "Attack")

    # Validation split comes only from official training data.
    fit_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=SEED, stratify=y
    )
    pre_val = make_preprocessor(X.iloc[fit_idx])
    X_fit = pre_val.fit_transform(X.iloc[fit_idx])
    X_val = pre_val.transform(X.iloc[val_idx])

    selection = []
    for cfg in CANDIDATES:
        model = LGBMClassifier(
            **cfg, max_depth=-1, colsample_bytree=0.9, subsample=0.9,
            class_weight="balanced", n_jobs=6, random_state=SEED, verbosity=-1,
        )
        start = time.perf_counter()
        model.fit(X_fit, y[fit_idx])
        pred = model.predict(X_val)
        elapsed = time.perf_counter() - start
        score = f1_score(y[val_idx], pred, average="macro")
        selection.append({"config": cfg, "validation_macro_f1": float(score),
                          "fit_and_validation_seconds": float(elapsed)})

    best = max(selection, key=lambda row: row["validation_macro_f1"])["config"]

    # Refit preprocessing and classifier on the complete official training split.
    pre = make_preprocessor(X)
    start = time.perf_counter()
    X_train_final = pre.fit_transform(X)
    X_test_final = pre.transform(X_test)
    model = LGBMClassifier(
        **best, max_depth=-1, colsample_bytree=0.9, subsample=0.9,
        class_weight="balanced", n_jobs=6, random_state=SEED, verbosity=-1,
    )
    model.fit(X_train_final, y)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = model.predict(X_test_final)
    inference_seconds = time.perf_counter() - start

    report = classification_report(y_test, prediction, labels=LABELS,
                                   output_dict=True, zero_division=0)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, prediction, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_test, prediction, labels=LABELS)
    metadata = {
        "dataset": "UNSW-NB15",
        "task": "binary intrusion detection",
        "target": "label (0=Normal, 1=Attack)",
        "feature_exclusions": ["id", "label", "attack_cat", "IP addresses", "timestamps"],
        "split": "official UNSW-NB15 training/testing CSV files",
        "train_records": int(len(y)),
        "test_records": int(len(y_test)),
        "feature_count": int(X_train_final.shape[1]),
        "model": "LightGBM",
        "selection_protocol": "20% stratified validation split from official training CSV only; best macro-F1 candidate selected; final model refit on full training CSV",
        "seed": SEED,
        "candidate_validation_results": selection,
        "selected_parameters": best,
        "test_metrics": {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "training_seconds_including_preprocessing": float(train_seconds),
            "inference_seconds": float(inference_seconds),
            "inference_ms_per_record": float(1000 * inference_seconds / len(y_test)),
            "per_class": report,
            "confusion_matrix": cm.tolist(),
            "labels": LABELS,
        },
        "warning": "This binary task is not directly comparable to the UNSW-NB15 10-class attack_cat task or NSL-KDD 5-class task.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "model": model, "metadata": metadata}, OUT / "model.joblib", compress=3)
    (OUT / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def pd_to_int(series):
    import pandas as pd
    return pd.to_numeric(series, errors="raise").astype(int).to_numpy()


if __name__ == "__main__":
    main()
