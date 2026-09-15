#!/usr/bin/env python3
"""
XGBoost baseline for next-visit CCSR prediction, evaluated under the exact
same schema as the reference BEHRT run and the other baselines in
scripts/baseline_comparison.py and scripts/logistic_regression_baseline.py
(same label vocabulary, same UNK exclusion, same validation-tuned decision
threshold procedure, same metric set, including the NEW vs RECURRING
breakdown).

This baseline reuses the EXACT SAME feature matrix as the logistic
regression baseline (history multi-hot + last-visit multi-hot + age +
visit_count) but replaces the linear model with a one-vs-rest gradient
boosted tree ensemble. The question this answers: is BEHRT's advantage
(concentrated in new-diagnosis detection, per the logistic regression
result) specific to sequence/attention modeling, or would a stronger
non-linear-but-still-non-sequential learner on the same flat features close
that gap too?

Model: one independent XGBoost binary classifier per label (hist tree
method, per-label scale_pos_weight to address the same extreme label
imbalance that the reference model addresses via pos_weight), trained via
sklearn's OneVsRestClassifier for parallelism across labels.

Usage:
    python3 scripts/xgboost_baseline.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from scripts.baseline_comparison import build_multihot, run_baseline
from scripts.logistic_regression_baseline import build_feature_matrix
from scripts.train_nextvisit_clean import (
    build_label_subset,
    build_label_support_counts,
    format_label_vocab,
)


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

    # Per-label scale_pos_weight (neg/pos ratio), same imbalance-correction
    # role as class_weight="balanced" in the logistic regression baseline.
    # OneVsRestClassifier clones the same estimator instance per label, so we
    # cannot pass per-label scale_pos_weight through it directly; instead we
    # rely on a single dataset-wide average ratio, which XGBoost tolerates
    # reasonably well when combined with hist tree method + shallow trees.
    pos_counts = train_y_true.sum(axis=0)
    neg_counts = train_y_true.shape[0] - pos_counts
    avg_scale_pos_weight = float(np.mean(neg_counts / np.maximum(pos_counts, 1)))
    print(f"  avg scale_pos_weight across labels = {avg_scale_pos_weight:.1f}", flush=True)

    print("Training one-vs-rest XGBoost (exact, shallow trees)...", flush=True)
    # NOTE: tree_method="hist" reliably segfaults on this platform/xgboost
    # build with this sparse feature matrix (reproduced via isolated smoke
    # tests: crashes regardless of scale_pos_weight, label positive-count, or
    # OneVsRestClassifier n_jobs). tree_method="exact" does not crash and
    # benchmarks at ~12s/label on the full training set, i.e. ~9-10 minutes
    # total across 470 labels with n_jobs=-1 on this 10-core machine.
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
    summary_xgb, per_class_xgb = run_baseline(
        "xgboost", val_y_prob, val_y_true, test_y_prob, test_y_true,
        label_vocab, train_label_supports, test_df, metric_exclude_labels=metric_exclude_labels,
    )
    print(f"  xgboost: APS={summary_xgb['metrics']['sample_wise_aps']:.4f} "
          f"AUC={summary_xgb['metrics']['sample_wise_auc']:.4f} "
          f"threshold={summary_xgb['threshold_tuning']['selected_threshold']:.2f}", flush=True)

    reference_metrics = {
        "run_id": "clean_run_20260910_124255",
        "sample_wise_aps": 0.2735172133816801,
        "sample_wise_auc": 0.9006313974430638,
        "micro_f1": 0.238104707408862,
        "selected_threshold": 0.6000000000000002,
    }

    # Pull in the logistic-regression + trivial-baseline results so the final
    # comparison table covers all models in one place.
    lr_path = results_dir / "logistic_regression_baseline.json"
    logistic_regression = {}
    trivial_baselines = {}
    if lr_path.exists():
        with open(lr_path, "r", encoding="utf-8") as f:
            lr_json = json.load(f)
        logistic_regression = lr_json.get("logistic_regression", {})
        trivial_baselines = lr_json.get("trivial_baselines", {})

    output = {
        "label_vocab_size": len(label_vocab),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_dim": int(train_x.shape[1]),
        "metric_exclude_labels": list(metric_exclude_labels),
        "avg_scale_pos_weight": avg_scale_pos_weight,
        "reference_model": reference_metrics,
        "xgboost": summary_xgb,
        "logistic_regression": logistic_regression,
        "trivial_baselines": trivial_baselines,
    }

    out_path = results_dir / "xgboost_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}", flush=True)

    csv_path = results_dir / "xgboost_per_class_metrics.csv"
    pd.DataFrame(per_class_xgb).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'Model':<22}{'APS':>10}{'AUC':>10}{'Tuned thr':>12}", flush=True)
    print(f"{'Reference (BEHRT)':<22}{reference_metrics['sample_wise_aps']:>10.4f}"
          f"{reference_metrics['sample_wise_auc']:>10.4f}{reference_metrics['selected_threshold']:>12.2f}",
          flush=True)
    print(f"{'xgboost':<22}{summary_xgb['metrics']['sample_wise_aps']:>10.4f}"
          f"{summary_xgb['metrics']['sample_wise_auc']:>10.4f}"
          f"{summary_xgb['threshold_tuning']['selected_threshold']:>12.2f}", flush=True)
    if logistic_regression:
        print(f"{'logistic_regression':<22}{logistic_regression['metrics']['sample_wise_aps']:>10.4f}"
              f"{logistic_regression['metrics']['sample_wise_auc']:>10.4f}"
              f"{logistic_regression['threshold_tuning']['selected_threshold']:>12.2f}", flush=True)
    for name, res in trivial_baselines.items():
        print(f"{name:<22}{res['metrics']['sample_wise_aps']:>10.4f}"
              f"{res['metrics']['sample_wise_auc']:>10.4f}"
              f"{res['threshold_tuning']['selected_threshold']:>12.2f}", flush=True)


if __name__ == "__main__":
    main()
