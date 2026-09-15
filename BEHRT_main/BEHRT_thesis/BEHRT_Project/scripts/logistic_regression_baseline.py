#!/usr/bin/env python3
"""
Logistic regression baseline for next-visit CCSR prediction, evaluated under
the exact same schema as the reference BEHRT run and the trivial baselines in
scripts/baseline_comparison.py (same label vocabulary, same UNK exclusion,
same validation-tuned decision threshold procedure, same metric set,
including the NEW vs RECURRING breakdown).

Unlike the trivial baselines (repeat_last_visit, popularity), this model DOES
use per-patient history, but only through simple, hand-engineered features
and a linear model -- no sequence modeling, no learned embeddings, no
attention. It isolates a specific question: does using this patient's own
history help at all beyond population base rates (popularity baseline), and
if so, does a transformer add value beyond what a simple linear model
extracts from the same history?

Features (per patient, at prediction time):
  - history_multihot: which CCSR codes have EVER appeared anywhere in the
    patient's visit history (multi-hot in the label vocabulary space).
  - last_visit_multihot: which CCSR codes appeared in the patient's most
    recent visit only (multi-hot; identical construction to the
    repeat_last_visit baseline's feature, reused as a recency signal).
  - age: the patient's age at the most recent visit (last entry of the age
    array).
  - visit_count: total number of prior visits (number of SEP-delimited
    segments in the raw code history).

Model: one independent logistic regression per label (liblinear solver,
L2-regularized, class_weight="balanced" to address the same extreme label
imbalance that the reference model addresses via pos_weight), trained via
sklearn's OneVsRestClassifier for parallelism across labels.

Usage:
    python3 scripts/logistic_regression_baseline.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from scripts.baseline_comparison import (
    NON_CODE_TOKENS,
    build_multihot,
    last_visit_codes,
    run_baseline,
)
from scripts.train_nextvisit_clean import (
    build_label_subset,
    build_label_support_counts,
    format_label_vocab,
    normalize_label_row,
)


def build_history_multihot(df, label_vocab):
    num_rows = len(df)
    num_labels = len(label_vocab)
    x = np.zeros((num_rows, num_labels), dtype=np.float32)
    for row_idx, history_codes in enumerate(df["code"]):
        for code in normalize_label_row(history_codes):
            if code in NON_CODE_TOKENS:
                continue
            idx = label_vocab.get(code)
            if idx is not None:
                x[row_idx, idx] = 1.0
    return x


def build_last_visit_multihot(df, label_vocab):
    num_rows = len(df)
    num_labels = len(label_vocab)
    x = np.zeros((num_rows, num_labels), dtype=np.float32)
    for row_idx, history_codes in enumerate(df["code"]):
        for code in last_visit_codes(history_codes):
            idx = label_vocab.get(code)
            if idx is not None:
                x[row_idx, idx] = 1.0
    return x


def build_scalar_features(df):
    num_rows = len(df)
    age = np.zeros((num_rows, 1), dtype=np.float32)
    visit_count = np.zeros((num_rows, 1), dtype=np.float32)
    for row_idx, (ages, codes) in enumerate(zip(df["age"], df["code"])):
        age[row_idx, 0] = float(ages[-1]) if len(ages) else 0.0
        codes_list = list(normalize_label_row(codes))
        visit_count[row_idx, 0] = float(sum(1 for c in codes_list if c == "SEP"))
    return age, visit_count


def build_feature_matrix(df, label_vocab, scaler_stats=None):
    history_x = build_history_multihot(df, label_vocab)
    last_visit_x = build_last_visit_multihot(df, label_vocab)
    age, visit_count = build_scalar_features(df)

    if scaler_stats is None:
        age_mean, age_std = float(age.mean()), float(age.std() or 1.0)
        visit_mean, visit_std = float(visit_count.mean()), float(visit_count.std() or 1.0)
        scaler_stats = {
            "age_mean": age_mean, "age_std": age_std,
            "visit_mean": visit_mean, "visit_std": visit_std,
        }
    age_scaled = (age - scaler_stats["age_mean"]) / scaler_stats["age_std"]
    visit_scaled = (visit_count - scaler_stats["visit_mean"]) / scaler_stats["visit_std"]

    x = sp.hstack([
        sp.csr_matrix(history_x),
        sp.csr_matrix(last_visit_x),
        sp.csr_matrix(age_scaled),
        sp.csr_matrix(visit_scaled),
    ]).tocsr()
    return x, scaler_stats


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

    print("Training one-vs-rest logistic regression (liblinear, class_weight=balanced)...", flush=True)
    t0 = time.time()
    clf = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=200,
        ),
        n_jobs=-1,
    )
    clf.fit(train_x, train_y_true)
    print(f"  Training done ({time.time() - t0:.1f}s)", flush=True)

    print("Scoring val/test...", flush=True)
    val_y_prob = clf.predict_proba(val_x).astype(np.float32)
    test_y_prob = clf.predict_proba(test_x).astype(np.float32)

    metric_exclude_labels = ("UNK",)
    summary_lr, per_class_lr = run_baseline(
        "logistic_regression", val_y_prob, val_y_true, test_y_prob, test_y_true,
        label_vocab, train_label_supports, test_df, metric_exclude_labels=metric_exclude_labels,
    )
    print(f"  logistic_regression: APS={summary_lr['metrics']['sample_wise_aps']:.4f} "
          f"AUC={summary_lr['metrics']['sample_wise_auc']:.4f} "
          f"threshold={summary_lr['threshold_tuning']['selected_threshold']:.2f}", flush=True)

    reference_metrics = {
        "run_id": "clean_run_20260910_124255",
        "sample_wise_aps": 0.2735172133816801,
        "sample_wise_auc": 0.9006313974430638,
        "micro_f1": 0.238104707408862,
        "selected_threshold": 0.6000000000000002,
    }

    # Load the trivial-baseline results so the final comparison table covers
    # all four models in one place.
    trivial_path = results_dir / "baseline_comparison.json"
    trivial_baselines = {}
    if trivial_path.exists():
        with open(trivial_path, "r", encoding="utf-8") as f:
            trivial_baselines = json.load(f).get("baselines", {})

    output = {
        "label_vocab_size": len(label_vocab),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "feature_dim": int(train_x.shape[1]),
        "metric_exclude_labels": list(metric_exclude_labels),
        "reference_model": reference_metrics,
        "logistic_regression": summary_lr,
        "trivial_baselines": trivial_baselines,
    }

    out_path = results_dir / "logistic_regression_baseline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}", flush=True)

    csv_path = results_dir / "logistic_regression_per_class_metrics.csv"
    pd.DataFrame(per_class_lr).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'Model':<22}{'APS':>10}{'AUC':>10}{'Tuned thr':>12}", flush=True)
    print(f"{'Reference (BEHRT)':<22}{reference_metrics['sample_wise_aps']:>10.4f}"
          f"{reference_metrics['sample_wise_auc']:>10.4f}{reference_metrics['selected_threshold']:>12.2f}",
          flush=True)
    print(f"{'logistic_regression':<22}{summary_lr['metrics']['sample_wise_aps']:>10.4f}"
          f"{summary_lr['metrics']['sample_wise_auc']:>10.4f}"
          f"{summary_lr['threshold_tuning']['selected_threshold']:>12.2f}", flush=True)
    for name, res in trivial_baselines.items():
        print(f"{name:<22}{res['metrics']['sample_wise_aps']:>10.4f}"
              f"{res['metrics']['sample_wise_auc']:>10.4f}"
              f"{res['threshold_tuning']['selected_threshold']:>12.2f}", flush=True)


if __name__ == "__main__":
    main()
