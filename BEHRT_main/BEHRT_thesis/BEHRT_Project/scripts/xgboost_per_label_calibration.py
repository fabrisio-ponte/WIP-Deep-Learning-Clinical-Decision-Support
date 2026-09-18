#!/usr/bin/env python3
"""
Per-label decision-threshold calibration for the XGBoost baseline.

The existing XGBoost baseline (scripts/xgboost_baseline.py) selects a single
global decision threshold (tuned on validation micro-F1) and applies it
uniformly across all ~470 CCSR labels. Because label prevalence varies by
orders of magnitude, a single global cutoff is a poor fit for most labels: it
under-predicts rare labels and may over/under-predict common ones. This shows
up as a large gap between XGBoost's threshold-independent ranking metrics
(sample-wise APS=0.489, AUC=0.931 -- both far better than BEHRT's
0.274/0.901) and its threshold-dependent micro-F1 (0.147, far worse than
BEHRT's 0.238): the model ranks well, but a single global threshold makes it
look far worse than it is at converting scores into decisions.

This script tunes ONE decision threshold PER LABEL (each maximizing that
label's own F1 on the validation set, with a fallback to the global
threshold for labels with zero positive examples in validation), then
recomputes test-set threshold-dependent metrics using that per-label
threshold vector, and reports the delta against the global-threshold
baseline. Ranking metrics (sample-wise APS/AUC) are threshold-independent
and are unaffected; only threshold-dependent metrics change.

Usage:
    python3 scripts/xgboost_per_label_calibration.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from scripts.baseline_comparison import build_multihot
from scripts.logistic_regression_baseline import build_feature_matrix
from scripts.train_nextvisit_clean import (
    build_label_subset,
    build_label_support_counts,
    compute_per_class_metrics,
    compute_top_k_metrics,
    evaluate_from_arrays,
    format_label_vocab,
    tune_decision_threshold,
)

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.05)


def tune_per_label_thresholds(y_true, y_prob, thresholds=THRESHOLD_GRID, fallback_threshold=0.5):
    """Pick, independently for each label column, the threshold in `thresholds`
    that maximizes that label's own F1 on (y_true, y_prob). Labels with zero
    positive examples fall back to `fallback_threshold` since F1 is always 0
    for them regardless of threshold, making tuning meaningless.
    """
    n_labels = y_true.shape[1]
    y_true_bool = y_true.astype(bool)
    support = y_true_bool.sum(axis=0)

    best_threshold = np.full(n_labels, float(fallback_threshold), dtype=np.float64)
    best_f1 = np.zeros(n_labels, dtype=np.float64)

    for t in thresholds:
        y_pred_bool = y_prob >= t
        tp = np.count_nonzero(y_true_bool & y_pred_bool, axis=0).astype(np.float64)
        fp = np.count_nonzero(~y_true_bool & y_pred_bool, axis=0).astype(np.float64)
        fn = np.count_nonzero(y_true_bool & ~y_pred_bool, axis=0).astype(np.float64)

        precision = np.divide(tp, tp + fp, out=np.zeros(n_labels), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros(n_labels), where=(tp + fn) > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_labels), where=denom > 0)

        improved = f1 > best_f1
        best_f1 = np.where(improved, f1, best_f1)
        best_threshold = np.where(improved, t, best_threshold)

    # Labels never seen positive in validation: tuning is meaningless (F1
    # stays 0 at every threshold), so keep the fallback threshold rather
    # than whatever arbitrary grid point happened to tie at 0.
    best_threshold = np.where(support > 0, best_threshold, float(fallback_threshold))
    return best_threshold, best_f1


def evaluate_per_label_threshold(y_true, y_prob, thresholds, label_vocab, train_supports, exclude_labels=("UNK",)):
    """Same metric set as train_nextvisit_clean.evaluate_from_arrays, but
    `thresholds` is a per-label array (one cutoff per column of
    y_true/y_prob) instead of a single scalar.
    """
    excluded_label_indices = set()
    metric_true = y_true
    metric_prob = y_prob
    metric_thresholds = thresholds

    if exclude_labels:
        excluded_label_indices = {
            int(label_vocab[label]) for label in exclude_labels if label in label_vocab
        }
        keep_indices = [idx for idx in range(y_true.shape[1]) if idx not in excluded_label_indices]
        metric_true = y_true[:, keep_indices]
        metric_prob = y_prob[:, keep_indices]
        metric_thresholds = thresholds[keep_indices]

    y_pred = (metric_prob >= metric_thresholds).astype(np.int64)

    aps = average_precision_score(metric_true, metric_prob, average="samples")
    try:
        auc = roc_auc_score(metric_true, metric_prob, average="samples")
    except ValueError:
        auc = float("nan")

    micro_precision = precision_score(metric_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(metric_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(metric_true, y_pred, average="micro", zero_division=0)
    samples_precision = precision_score(metric_true, y_pred, average="samples", zero_division=0)
    samples_recall = recall_score(metric_true, y_pred, average="samples", zero_division=0)
    samples_f1 = f1_score(metric_true, y_pred, average="samples", zero_division=0)
    subset_accuracy = float(np.mean(np.all(metric_true == y_pred, axis=1)))
    hamming_accuracy = 1.0 - hamming_loss(metric_true, y_pred)
    top_k_metrics = compute_top_k_metrics(metric_true, metric_prob, top_k_values=(5, 10))

    full_pred = (y_prob >= thresholds).astype(np.int64)
    per_class_metrics = compute_per_class_metrics(
        y_true, y_prob, full_pred, label_vocab, train_supports,
        excluded_label_indices=excluded_label_indices,
    )

    metrics = {
        "sample_wise_aps": float(aps),
        "sample_wise_auc": float(auc),
        "subset_accuracy": float(subset_accuracy),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "samples_precision": float(samples_precision),
        "samples_recall": float(samples_recall),
        "samples_f1": float(samples_f1),
        "hamming_accuracy": float(hamming_accuracy),
        "top_k_metrics": top_k_metrics,
    }
    return metrics, per_class_metrics


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_nextvisit_ccsr_clean.parquet"
    val_path = data_dir / "val_nextvisit_ccsr_clean.parquet"
    test_path = data_dir / "test_nextvisit_ccsr_clean.parquet"
    vocab_path = data_dir / "vocab_ccsr_clean"

    print("Loading data...", flush=True)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    bert_vocab = load_obj(str(vocab_path))
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])

    label_vocab, train_df, label_strategy = build_label_subset(
        train_df, base_label_vocab, top_k_labels=0, min_label_freq=0.0
    )
    print(f"Label vocab size: {len(label_vocab)} (strategy={label_strategy})", flush=True)

    train_label_supports = build_label_support_counts(train_df, label_vocab)

    print("Building ground-truth multi-hot targets...", flush=True)
    train_y_true = build_multihot(train_df["label"], label_vocab)
    val_y_true = build_multihot(val_df["label"], label_vocab)
    test_y_true = build_multihot(test_df["label"], label_vocab)

    print("Building feature matrices (history multi-hot + last-visit multi-hot + age + visit_count)...", flush=True)
    t0 = time.time()
    train_x, scaler_stats = build_feature_matrix(train_df, label_vocab)
    val_x, _ = build_feature_matrix(val_df, label_vocab, scaler_stats=scaler_stats)
    test_x, _ = build_feature_matrix(test_df, label_vocab, scaler_stats=scaler_stats)
    print(f"  train_x shape={train_x.shape} val_x shape={val_x.shape} test_x shape={test_x.shape} "
          f"({time.time() - t0:.1f}s)", flush=True)

    pos_counts = train_y_true.sum(axis=0)
    neg_counts = train_y_true.shape[0] - pos_counts
    avg_scale_pos_weight = float(np.mean(neg_counts / np.maximum(pos_counts, 1)))
    print(f"  avg scale_pos_weight across labels = {avg_scale_pos_weight:.1f}", flush=True)

    print("Training one-vs-rest XGBoost (exact, shallow trees; same hyperparameters as "
          "scripts/xgboost_baseline.py)...", flush=True)
    t0 = time.time()
    base_clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.3,
        tree_method="exact",
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=avg_scale_pos_weight,
        n_jobs=1,
        verbosity=0,
    )
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    clf.fit(train_x, train_y_true)
    print(f"  Training done ({time.time() - t0:.1f}s)", flush=True)

    print("Scoring val/test...", flush=True)
    val_y_prob = clf.predict_proba(val_x).astype(np.float32)
    test_y_prob = clf.predict_proba(test_x).astype(np.float32)

    metric_exclude_labels = ("UNK",)

    print("Tuning GLOBAL threshold (single cutoff for all labels, val micro-F1)...", flush=True)
    global_threshold, global_val_score, _ = tune_decision_threshold(
        val_y_true, val_y_prob, metric_name="micro_f1",
        threshold_min=0.1, threshold_max=0.9, threshold_step=0.05,
    )
    print(f"  global_threshold={global_threshold:.2f} (val micro_f1={global_val_score:.4f})", flush=True)

    global_test_metrics = evaluate_from_arrays(
        test_y_true, test_y_prob, threshold=global_threshold,
        label_vocab=label_vocab, train_supports=train_label_supports,
        include_per_class=False, exclude_labels=metric_exclude_labels,
    )

    print("Tuning PER-LABEL thresholds (one cutoff per label, val per-label F1)...", flush=True)
    per_label_thresholds, per_label_val_f1 = tune_per_label_thresholds(
        val_y_true, val_y_prob, fallback_threshold=global_threshold,
    )
    n_fallback = int(np.sum(val_y_true.sum(axis=0) == 0))
    print(f"  {n_fallback}/{len(label_vocab)} labels had zero positives in validation "
          f"(fell back to global threshold={global_threshold:.2f})", flush=True)

    per_label_test_metrics, per_label_per_class = evaluate_per_label_threshold(
        test_y_true, test_y_prob, per_label_thresholds, label_vocab, train_label_supports,
        exclude_labels=metric_exclude_labels,
    )

    print(f"  global-threshold : APS={global_test_metrics['sample_wise_aps']:.4f} "
          f"AUC={global_test_metrics['sample_wise_auc']:.4f} "
          f"micro_F1={global_test_metrics['micro_f1']:.4f}", flush=True)
    print(f"  per-label-thresh : APS={per_label_test_metrics['sample_wise_aps']:.4f} "
          f"AUC={per_label_test_metrics['sample_wise_auc']:.4f} "
          f"micro_F1={per_label_test_metrics['micro_f1']:.4f}", flush=True)

    reference_metrics = {
        "run_id": "clean_run_20260910_124255",
        "sample_wise_aps": 0.2735172133816801,
        "sample_wise_auc": 0.9006313974430638,
        "micro_f1": 0.238104707408862,
        "selected_threshold": 0.6000000000000002,
    }

    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    per_label_threshold_table = [
        {
            "label": idx_to_label[idx],
            "threshold": float(per_label_thresholds[idx]),
            "val_f1_at_threshold": float(per_label_val_f1[idx]),
            "val_support": int(val_y_true[:, idx].sum()),
        }
        for idx in range(len(label_vocab))
        if idx_to_label[idx] not in metric_exclude_labels
    ]

    output = {
        "label_vocab_size": len(label_vocab),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "metric_exclude_labels": list(metric_exclude_labels),
        "avg_scale_pos_weight": avg_scale_pos_weight,
        "reference_model": reference_metrics,
        "global_threshold": {
            "threshold": global_threshold,
            "val_micro_f1": global_val_score,
            "test_metrics": global_test_metrics,
        },
        "per_label_threshold": {
            "n_labels_fallback_to_global": n_fallback,
            "test_metrics": per_label_test_metrics,
            "threshold_table": per_label_threshold_table,
        },
    }

    out_path = results_dir / "xgboost_per_label_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}", flush=True)

    csv_path = results_dir / "xgboost_per_label_calibrated_per_class_metrics.csv"
    pd.DataFrame(per_label_per_class).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    print("\n=== Summary (test set) ===", flush=True)
    print(f"{'Strategy':<22}{'APS':>10}{'AUC':>10}{'micro_F1':>10}", flush=True)
    print(f"{'Reference (BEHRT)':<22}{reference_metrics['sample_wise_aps']:>10.4f}"
          f"{reference_metrics['sample_wise_auc']:>10.4f}{reference_metrics['micro_f1']:>10.4f}", flush=True)
    print(f"{'xgboost (global thr)':<22}{global_test_metrics['sample_wise_aps']:>10.4f}"
          f"{global_test_metrics['sample_wise_auc']:>10.4f}{global_test_metrics['micro_f1']:>10.4f}", flush=True)
    print(f"{'xgboost (per-label)':<22}{per_label_test_metrics['sample_wise_aps']:>10.4f}"
          f"{per_label_test_metrics['sample_wise_auc']:>10.4f}{per_label_test_metrics['micro_f1']:>10.4f}", flush=True)


if __name__ == "__main__":
    main()
