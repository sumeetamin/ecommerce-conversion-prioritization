"""Run leakage-aware model selection on UCI Online Shoppers Purchasing Intention."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmark_data"
ZIP_PATH = DATA_DIR / "online_shoppers.zip"
CSV_PATH = DATA_DIR / "online_shoppers_intention.csv"
ZIP_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
OUTPUT = ROOT / "data/benchmark.json"
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        if not ZIP_PATH.exists():
            request = urllib.request.Request(ZIP_URL, headers={"User-Agent": "ecommerce-conversion-prioritization/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, ZIP_PATH.open("wb") as output:
                output.write(response.read())
        with zipfile.ZipFile(ZIP_PATH) as archive:
            member = next(name for name in archive.namelist() if name.endswith(".csv"))
            CSV_PATH.write_bytes(archive.read(member))


def model_pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("features", prep), ("model", model)])


def metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = prob >= threshold
    return {
        "average_precision": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def threshold_row(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = prob >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    n = len(y)
    return {
        "threshold": round(float(threshold), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "sessions_reviewed": tp + fp, "review_share": (tp + fp) / max(n, 1),
        "conversion_capture_share": tp / max(int(y.sum()), 1),
    }


def main() -> None:
    ensure_data()
    started = time.perf_counter()
    frame = pd.read_csv(CSV_PATH)
    target = frame.pop("Revenue").astype(bool).astype(int).to_numpy()
    categorical = ["OperatingSystems", "Browser", "Region", "TrafficType", "Month", "VisitorType", "Weekend"]
    numeric = [column for column in frame.columns if column not in categorical]
    x_train, x_temp, y_train, y_temp = train_test_split(frame, target, test_size=0.40, stratify=target, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    candidates = {
        "logistic_regression": model_pipeline(LogisticRegression(C=0.5, max_iter=2500, random_state=SEED), numeric, categorical),
        "hist_gradient_boosting": model_pipeline(HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED,
        ), numeric, categorical),
    }
    validation: dict[str, dict[str, float]] = {}
    baseline_val = np.repeat(float(y_train.mean()), len(y_val))
    validation["most_frequent_baseline"] = metrics(y_val, baseline_val)
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        validation[name] = metrics(y_val, pipeline.predict_proba(x_val)[:, 1])
    selected = max(candidates, key=lambda name: validation[name]["average_precision"])

    val_prob = candidates[selected].predict_proba(x_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = max(thresholds, key=lambda threshold: f1_score(y_val, val_prob >= threshold, zero_division=0))
    x_fit = pd.concat([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val])
    final_model = model_pipeline(
        LogisticRegression(C=0.5, max_iter=2500, random_state=SEED) if selected == "logistic_regression" else HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED,
        ), numeric, categorical,
    )
    final_model.fit(x_fit, y_fit)
    test_prob = final_model.predict_proba(x_test)[:, 1]
    baseline_test_prob = np.repeat(float(y_fit.mean()), len(y_test))
    report = {
        "project": "E-commerce Conversion Prioritization Lab",
        "dataset": {
            "name": "Online Shoppers Purchasing Intention Dataset", "source": "UCI Machine Learning Repository",
            "source_url": "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
            "citation": "Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI ML Repository. DOI: 10.24432/C5F88Q.",
            "license": "CC BY 4.0", "rows": int(len(frame)), "features": int(frame.shape[1]),
            "positive_rate": float(target.mean()), "archive_sha256": sha256(ZIP_PATH),
        },
        "protocol": {
            "split": "stratified 60/20/20 train/validation/test; random_state=42",
            "selection_metric": "validation average precision", "selected_model": selected,
            "candidate_validation_metrics": validation,
            "threshold_selection": "maximum validation F1 over 0.05–0.95; evaluated once on held-out test",
            "threshold": float(best_threshold), "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "fit_seconds": round(time.perf_counter() - started, 3),
            "feature_timing": "end-of-session features; not an early-session intervention model",
        },
        "test": {
            "sessions": int(len(y_test)), "conversions": int(y_test.sum()),
            "prevalence": float(y_test.mean()), "metrics_at_0_5": metrics(y_test, test_prob, 0.5),
            "majority_probability_baseline": metrics(y_test, baseline_test_prob, 0.5),
            "metrics_at_validation_threshold": metrics(y_test, test_prob, float(best_threshold)),
            "threshold_curve": [threshold_row(y_test, test_prob, threshold) for threshold in sorted(set([*thresholds, best_threshold]))],
            "visitor_segments": [
                {"visitor_type": str(group), "sessions": int(len(indices)), "conversion_rate": float(y_test[indices].mean()),
                 "mean_score": float(test_prob[indices].mean())}
                for group in sorted(x_test["VisitorType"].unique())
                for indices in [np.flatnonzero(x_test["VisitorType"].to_numpy() == group)] if len(indices)
            ],
        },
        "limitations": [
            "A predictive score does not estimate the causal lift of contacting a visitor.",
            "Features describe session behavior and are appropriate only for end-of-session prioritization.",
            "The random holdout is not a forward-in-time deployment simulation; this benchmark is a portfolio demonstration.",
            "The data is historical, from one store, and has a 15.5% positive-class rate; results may not transfer to another business.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rows={len(frame):,}; test={len(y_test):,}; positive_rate={target.mean():.3%}")
    print(f"Selected={selected}; validation AP={validation[selected]['average_precision']:.4f}")
    print(f"Test AP={report['test']['metrics_at_validation_threshold']['average_precision']:.4f}; ROC-AUC={report['test']['metrics_at_validation_threshold']['roc_auc']:.4f}; Brier={report['test']['metrics_at_validation_threshold']['brier_score']:.4f}")
    print(f"Threshold={best_threshold:.3f}; precision={report['test']['metrics_at_validation_threshold']['precision']:.3f}; recall={report['test']['metrics_at_validation_threshold']['recall']:.3f}")
    print(f"Aggregate report: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

