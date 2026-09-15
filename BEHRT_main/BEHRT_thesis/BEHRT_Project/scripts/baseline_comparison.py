#!/usr/bin/env python3
"""
Non-neural baselines for next-visit CCSR prediction, evaluated under the exact
same schema as the reference BEHRT run (train_nextvisit_clean.py): same
label vocabulary, same UNK exclusion, same validation-tuned decision
threshold procedure, same metric set (sample-wise AUC/APS, micro/samples
precision-recall-F1, top-k, per-class metrics, NEW vs RECURRING breakdown).

Two baselines are computed, both requiring no model training:

1. "repeat_last_visit": predicts exactly the set of CCSR codes present at the
   patient's most recent visit (score=1.0 for codes in that visit, 0.0
   otherwise). Tests how much of the reference model's performance is
   explained by simply copying the last visit forward, given the project's
   own finding that 70% of next-visit targets are recurrences.

2. "popularity": predicts the same fixed ranking of CCSR codes for every
   patient, ranked/scored by their overall training-set frequency. Tests how
   much performance is explained purely by exploiting class imbalance/base
   rates rather than any per-patient signal.

Usage:
    python3 scripts/baseline_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from scripts.train_nextvisit_clean import (
    build_label_subset,
    build_label_support_counts,
    compute_new_recurring_metrics,
    compute_recurring_mask,
    evaluate_from_arrays,
    format_label_vocab,
    normalize_label_row,
    tune_decision_threshold,
)

NON_CODE_TOKENS = {"PAD", "SEP", "CLS", "MASK"}


def build_multihot(df_labels, label_vocab):
    num_rows = len(df_labels)
    num_labels = len(label_vocab)
    y = np.zeros((num_rows, num_labels), dtype=np.int64)
    for row_idx, labels in enumerate(df_labels):
        for code in normalize_label_row(labels):
            idx = label_vocab.get(code)
            if idx is not None:
                y[row_idx, idx] = 1
    return y


def last_visit_codes(history_codes):
    codes = list(normalize_label_row(history_codes))
    sep_positions = [i for i, c in enumerate(codes) if c == "SEP"]
    if not sep_positions:
        segment = codes
    else:
        last_sep = sep_positions[-1]
        prev_sep = sep_positions[-2] if len(sep_positions) >= 2 else -1
        segment = codes[prev_sep + 1 : last_sep]
    return set(c for c in segment if c not in NON_CODE_TOKENS)


def build_repeat_last_visit_scores(df, label_vocab):
    num_rows = len(df)
    num_labels = len(label_vocab)
    y_prob = np.zeros((num_rows, num_labels), dtype=np.float32)
    for row_idx, history_codes in enumerate(df["code"]):
        for code in last_visit_codes(history_codes):
            idx = label_vocab.get(code)
            if idx is not None:
                y_prob[row_idx, idx] = 1.0
    return y_prob


def build_popularity_scores(num_rows, label_vocab, train_label_supports, num_train_rows):
    num_labels = len(label_vocab)
    freq_vector = np.zeros(num_labels, dtype=np.float64)
    for label, idx in label_vocab.items():
        freq_vector[idx] = train_label_supports.get(label, 0) / max(num_train_rows, 1)
    y_prob = np.tile(freq_vector, (num_rows, 1)).astype(np.float32)
    return y_prob


def run_baseline(name, val_y_prob, val_y_true, test_y_prob, test_y_true, label_vocab, train_label_supports, test_df,
                  metric_exclude_labels=("UNK",), threshold_metric="micro_f1",
                  threshold_min=0.1, threshold_max=0.9, threshold_step=0.05):
    tuned_threshold, tuned_score, threshold_history = tune_decision_threshold(
        val_y_true,
        val_y_prob,
        metric_name=threshold_metric,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
    )

    test_metrics = evaluate_from_arrays(
        test_y_true,
        test_y_prob,
        threshold=tuned_threshold,
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        include_per_class=True,
        exclude_labels=metric_exclude_labels,
    )
    test_metrics_0_5 = evaluate_from_arrays(
        test_y_true,
        test_y_prob,
        threshold=0.5,
        label_vocab=label_vocab,
        train_supports=train_label_supports,
        include_per_class=False,
        exclude_labels=metric_exclude_labels,
    )

    recurring_mask_full = compute_recurring_mask(test_df, label_vocab)
    metric_true_nr = test_y_true
    metric_prob_nr = test_y_prob
    metric_mask_nr = recurring_mask_full
    if metric_exclude_labels:
        excluded_indices_nr = {
            int(label_vocab[label]) for label in metric_exclude_labels if label in label_vocab
        }
        keep_indices_nr = [idx for idx in range(test_y_true.shape[1]) if idx not in excluded_indices_nr]
        metric_true_nr = test_y_true[:, keep_indices_nr]
        metric_prob_nr = test_y_prob[:, keep_indices_nr]
        metric_mask_nr = recurring_mask_full[:, keep_indices_nr]

    new_vs_recurring = compute_new_recurring_metrics(metric_true_nr, metric_prob_nr, tuned_threshold, metric_mask_nr)

    per_class_metrics = test_metrics.pop("per_class_metrics")

    return {
        "name": name,
        "threshold_tuning": {
            "metric": threshold_metric,
            "selected_threshold": tuned_threshold,
            "selected_score": tuned_score,
        },
        "metrics_threshold_0_5": test_metrics_0_5,
        "metrics": test_metrics,
        "new_vs_recurring_metrics": new_vs_recurring,
        "num_per_class_rows": len(per_class_metrics),
    }, per_class_metrics


def main():
    data_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_nextvisit_ccsr_clean.parquet"
    val_path = data_dir / "val_nextvisit_ccsr_clean.parquet"
    test_path = data_dir / "test_nextvisit_ccsr_clean.parquet"
    vocab_path = data_dir / "vocab_ccsr_clean"

    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    bert_vocab = load_obj(str(vocab_path))
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])

    # Match the reference run's label-vocab strategy: no top-k/min-freq filtering
    # (full 463-label vocabulary).
    label_vocab, train_df, label_strategy = build_label_subset(
        train_df, base_label_vocab, top_k_labels=0, min_label_freq=0.0
    )
    print(f"Label vocab size: {len(label_vocab)} (strategy={label_strategy})")

    train_label_supports = build_label_support_counts(train_df, label_vocab)

    print("Building ground-truth multi-hot targets...")
    val_y_true = build_multihot(val_df["label"], label_vocab)
    test_y_true = build_multihot(test_df["label"], label_vocab)

    metric_exclude_labels = ("UNK",)

    all_results = {}
    all_per_class = {}

    print("Building repeat_last_visit scores...")
    val_prob_rlv = build_repeat_last_visit_scores(val_df, label_vocab)
    test_prob_rlv = build_repeat_last_visit_scores(test_df, label_vocab)
    summary_rlv, per_class_rlv = run_baseline(
        "repeat_last_visit", val_prob_rlv, val_y_true, test_prob_rlv, test_y_true,
        label_vocab, train_label_supports, test_df, metric_exclude_labels=metric_exclude_labels,
    )
    all_results["repeat_last_visit"] = summary_rlv
    all_per_class["repeat_last_visit"] = per_class_rlv
    print(f"  repeat_last_visit: APS={summary_rlv['metrics']['sample_wise_aps']:.4f} "
          f"AUC={summary_rlv['metrics']['sample_wise_auc']:.4f} "
          f"threshold={summary_rlv['threshold_tuning']['selected_threshold']:.2f}")

    print("Building popularity scores...")
    val_prob_pop = build_popularity_scores(len(val_df), label_vocab, train_label_supports, len(train_df))
    test_prob_pop = build_popularity_scores(len(test_df), label_vocab, train_label_supports, len(train_df))
    summary_pop, per_class_pop = run_baseline(
        "popularity", val_prob_pop, val_y_true, test_prob_pop, test_y_true,
        label_vocab, train_label_supports, test_df, metric_exclude_labels=metric_exclude_labels,
    )
    all_results["popularity"] = summary_pop
    all_per_class["popularity"] = per_class_pop
    print(f"  popularity: APS={summary_pop['metrics']['sample_wise_aps']:.4f} "
          f"AUC={summary_pop['metrics']['sample_wise_auc']:.4f} "
          f"threshold={summary_pop['threshold_tuning']['selected_threshold']:.2f}")

    reference_metrics = {
        "run_id": "clean_run_20260910_124255",
        "sample_wise_aps": 0.2735172133816801,
        "sample_wise_auc": 0.9006313974430638,
        "micro_f1": 0.238104707408862,
        "selected_threshold": 0.6000000000000002,
    }

    output = {
        "label_vocab_size": len(label_vocab),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "metric_exclude_labels": list(metric_exclude_labels),
        "reference_model": reference_metrics,
        "baselines": all_results,
    }

    out_path = results_dir / "baseline_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")

    for name, per_class in all_per_class.items():
        csv_path = results_dir / f"baseline_{name}_per_class_metrics.csv"
        pd.DataFrame(per_class).to_csv(csv_path, index=False)
        print(f"Wrote {csv_path}")

    print("\n=== Summary ===")
    print(f"{'Model':<20}{'APS':>10}{'AUC':>10}{'Tuned thr':>12}")
    print(f"{'Reference (BEHRT)':<20}{reference_metrics['sample_wise_aps']:>10.4f}"
          f"{reference_metrics['sample_wise_auc']:>10.4f}{reference_metrics['selected_threshold']:>12.2f}")
    for name, res in all_results.items():
        print(f"{name:<20}{res['metrics']['sample_wise_aps']:>10.4f}"
              f"{res['metrics']['sample_wise_auc']:>10.4f}"
              f"{res['threshold_tuning']['selected_threshold']:>12.2f}")


if __name__ == "__main__":
    main()
