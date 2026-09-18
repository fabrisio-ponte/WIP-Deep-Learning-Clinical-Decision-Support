#!/usr/bin/env python3
"""
Bootstrap confidence intervals on test-set metrics and on the metric deltas
between the reference BEHRT model and each baseline (popularity,
repeat_last_visit, logistic_regression, xgboost).

Addresses a gap flagged during the publication-readiness review: every
metric reported so far (APS, AUC, micro-F1) is a single point estimate on
one fixed test split, with no uncertainty quantification. Without this,
"BEHRT beats popularity by +7.5% relative APS" cannot be distinguished from
sampling noise.

Method: fix each model's already-published decision threshold and test-set
predictions (re-computed here, since raw prediction arrays were not
persisted by the original baseline scripts), then bootstrap-resample TEST
ROWS (patients) with replacement, recomputing sample-wise APS, sample-wise
AUC, and micro-F1 on each resample. This is a percentile bootstrap over the
fixed test set (captures sampling variability of the *evaluation* set, not
full retraining variability) -- the standard, cheap approach used when
retraining per bootstrap replicate is infeasible.

For BEHRT vs each baseline, the SAME bootstrap row indices are reused for
both models per replicate (paired bootstrap), giving a proper distribution
of the paired metric delta and a two-sided bootstrap p-value (fraction of
replicates where the delta's sign flips relative to the observed delta).

Usage:
    python3 scripts/bootstrap_confidence_intervals.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction
from scripts.baseline_comparison import (
    build_multihot,
    build_popularity_scores,
    build_repeat_last_visit_scores,
)
from scripts.logistic_regression_baseline import build_feature_matrix
from scripts.train_nextvisit_clean import (
    BertConfig,
    build_label_subset,
    build_label_support_counts,
    collect_eval_arrays,
    format_label_vocab,
)

N_BOOT = 500
RNG_SEED = 42
REFERENCE_RUN_ID = "clean_run_20260910_124255"
REFERENCE_THRESHOLD = 0.6000000000000002


def load_json_threshold(path, key_path):
    with open(path) as f:
        data = json.load(f)
    node = data
    for key in key_path:
        node = node[key]
    return float(node)


def get_behrt_test_probs(data_dir, project_root, label_vocab, bert_vocab, test_df):
    run_dir = project_root / "data" / "models" / "clean_runs" / REFERENCE_RUN_ID
    checkpoint_path = run_dir / "behrt_nextvisit_ccsr_clean_best.pt"

    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)
    global_params = {
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_len_seq": 64,
    }
    model_config = {
        "vocab_size": len(bert_vocab["token2idx"]),
        "hidden_size": 288,
        "seg_vocab_size": 2,
        "age_vocab_size": len(age_vocab_dict),
        "max_position_embedding": global_params["max_len_seq"],
        "hidden_dropout_prob": 0.1,
        "num_hidden_layers": 6,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1,
        "intermediate_size": 256,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
    }
    feature_dict = {"word": True, "seg": True, "age": True, "position": True}

    mlb = MultiLabelBinarizer(classes=list(label_vocab.values()))
    mlb.fit([[x] for x in list(label_vocab.values())])

    conf = BertConfig(model_config)
    model = BertForMultiLabelPrediction(conf, num_labels=len(label_vocab), feature_dict=feature_dict)
    model.load_state_dict(torch.load(checkpoint_path, map_location=global_params["device"]))
    model = model.to(global_params["device"])

    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"], label2idx=label_vocab, age2idx=age_vocab_dict,
        dataframe=test_df, max_len=global_params["max_len_seq"],
    )
    test_loader = DataLoader(test_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)

    test_y_true, test_y_prob = collect_eval_arrays(model, test_loader, mlb, global_params["device"])
    return test_y_true, test_y_prob


def bootstrap_metrics(y_true, y_prob, threshold, boot_indices):
    """Vectorized-per-replicate bootstrap of sample-wise APS/AUC and micro-F1.
    Deliberately skips top-k/per-class metrics (expensive Python loops) since
    only the three headline metrics are needed here."""
    aps_vals = np.empty(len(boot_indices))
    auc_vals = np.empty(len(boot_indices))
    f1_vals = np.empty(len(boot_indices))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, idx in enumerate(boot_indices):
            bt, bp = y_true[idx], y_prob[idx]
            bpred = (bp >= threshold).astype(np.int64)
            aps_vals[i] = average_precision_score(bt, bp, average="samples")
            try:
                auc_vals[i] = roc_auc_score(bt, bp, average="samples")
            except ValueError:
                auc_vals[i] = np.nan
            f1_vals[i] = f1_score(bt, bpred, average="micro", zero_division=0)
    return aps_vals, auc_vals, f1_vals


def summarize(name, point_true, point_prob, threshold, boot_indices):
    aps_vals, auc_vals, f1_vals = bootstrap_metrics(point_true, point_prob, threshold, boot_indices)
    point_pred = (point_prob >= threshold).astype(np.int64)
    point_aps = average_precision_score(point_true, point_prob, average="samples")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point_auc = roc_auc_score(point_true, point_prob, average="samples")
    point_f1 = f1_score(point_true, point_pred, average="micro", zero_division=0)

    def ci(vals):
        clean = vals[~np.isnan(vals)]
        return [float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))]

    result = {
        "name": name,
        "threshold": float(threshold),
        "sample_wise_aps": {"point": float(point_aps), "ci95": ci(aps_vals)},
        "sample_wise_auc": {"point": float(point_auc), "ci95": ci(auc_vals)},
        "micro_f1": {"point": float(point_f1), "ci95": ci(f1_vals)},
    }
    print(f"  {name:<20} APS={point_aps:.4f} {ci(aps_vals)}  "
          f"AUC={point_auc:.4f} {ci(auc_vals)}  F1={point_f1:.4f} {ci(f1_vals)}", flush=True)
    return result, {"aps": aps_vals, "auc": auc_vals, "f1": f1_vals}


def paired_delta(name_a, name_b, boot_a, boot_b, point_a, point_b, metric_key):
    delta_obs = point_a[metric_key] - point_b[metric_key]
    deltas = boot_a[metric_key] - boot_b[metric_key]
    deltas = deltas[~np.isnan(deltas)]
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p-value: fraction of replicates with sign opposite to observed delta, doubled
    if delta_obs >= 0:
        p_value = 2.0 * np.mean(deltas <= 0)
    else:
        p_value = 2.0 * np.mean(deltas >= 0)
    p_value = float(min(p_value, 1.0))
    return {
        "comparison": f"{name_a} - {name_b}",
        "metric": metric_key,
        "delta": float(delta_obs),
        "ci95": [float(ci_low), float(ci_high)],
        "bootstrap_p_value": p_value,
        "significant_at_0.05": bool(ci_low > 0 or ci_high < 0),
    }


def main():
    t_start = time.time()
    data_dir = PROJECT_ROOT / "data" / "processed"
    results_dir = PROJECT_ROOT / "results"

    print("Loading data...", flush=True)
    train_df = pd.read_parquet(data_dir / "train_nextvisit_ccsr_clean.parquet")
    test_df = pd.read_parquet(data_dir / "test_nextvisit_ccsr_clean.parquet")
    bert_vocab = load_obj(str(data_dir / "vocab_ccsr_clean"))
    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])
    label_vocab, train_df, _ = build_label_subset(train_df, base_label_vocab, top_k_labels=0, min_label_freq=0.0)
    train_label_supports = build_label_support_counts(train_df, label_vocab)
    print(f"Label vocab size: {len(label_vocab)}", flush=True)

    test_y_true = build_multihot(test_df["label"], label_vocab)

    unk_idx = label_vocab.get("UNK")
    exclude_cols = {unk_idx} if unk_idx is not None else set()
    keep_cols = [i for i in range(len(label_vocab)) if i not in exclude_cols]

    rng = np.random.default_rng(RNG_SEED)
    n_test = len(test_df)
    boot_indices = [rng.integers(0, n_test, size=n_test) for _ in range(N_BOOT)]

    predictions = {}
    thresholds = {}

    # --- Popularity & repeat_last_visit (cheap, deterministic) ---
    print("Computing popularity / repeat_last_visit predictions...", flush=True)
    predictions["popularity"] = build_popularity_scores(len(test_df), label_vocab, train_label_supports, len(train_df))
    predictions["repeat_last_visit"] = build_repeat_last_visit_scores(test_df, label_vocab)
    thresholds["popularity"] = load_json_threshold(
        results_dir / "baseline_comparison.json",
        ["baselines", "popularity", "threshold_tuning", "selected_threshold"])
    thresholds["repeat_last_visit"] = load_json_threshold(
        results_dir / "baseline_comparison.json",
        ["baselines", "repeat_last_visit", "threshold_tuning", "selected_threshold"])

    # --- Logistic regression (refit, fast) ---
    print("Refitting logistic regression...", flush=True)
    t0 = time.time()
    train_x, scaler_stats = build_feature_matrix(train_df, label_vocab)
    test_x, _ = build_feature_matrix(test_df, label_vocab, scaler_stats=scaler_stats)
    train_y_true = build_multihot(train_df["label"], label_vocab)
    lr_clf = OneVsRestClassifier(
        LogisticRegression(penalty="l2", solver="liblinear", class_weight="balanced", max_iter=200),
        n_jobs=-1,
    )
    lr_clf.fit(train_x, train_y_true)
    predictions["logistic_regression"] = lr_clf.predict_proba(test_x).astype(np.float32)
    thresholds["logistic_regression"] = load_json_threshold(
        results_dir / "logistic_regression_baseline.json",
        ["logistic_regression", "threshold_tuning", "selected_threshold"])
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    # --- XGBoost (load cached test predictions instead of refitting) ---
    # Refitting XGBoost in-process here segfaulted twice (SIGSEGV in
    # libxgboost.dylib) under high swap pressure, even after reducing
    # OneVsRestClassifier n_jobs from -1 to 3 -- likely because this script
    # already holds train/test feature matrices + a fitted LR model in memory
    # by this point. Instead, reuse the cached test_y_prob written by
    # xgboost_baseline.py (run standalone, which has fit successfully before).
    print("Loading cached XGBoost test predictions...", flush=True)
    t0 = time.time()
    xgb_prob_path = results_dir / "xgboost_test_y_prob.npy"
    if not xgb_prob_path.exists():
        raise FileNotFoundError(
            f"{xgb_prob_path} not found -- run scripts/xgboost_baseline.py first "
            "(it now caches test_y_prob for reuse by this script).")
    xgb_test_prob = np.load(xgb_prob_path)
    if xgb_test_prob.shape != (len(test_df), len(label_vocab)):
        raise ValueError(
            f"Cached xgboost_test_y_prob shape {xgb_test_prob.shape} does not match "
            f"expected ({len(test_df)}, {len(label_vocab)}) -- label_vocab or test "
            "split may have changed since xgboost_baseline.py was last run.")
    predictions["xgboost"] = xgb_test_prob.astype(np.float32)
    thresholds["xgboost"] = load_json_threshold(
        results_dir / "xgboost_baseline.json", ["xgboost", "threshold_tuning", "selected_threshold"])
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    # --- BEHRT reference (inference only, no threshold sweep -- known threshold reused) ---
    print("Running BEHRT reference checkpoint inference on test set...", flush=True)
    t0 = time.time()
    behrt_y_true, behrt_y_prob = get_behrt_test_probs(data_dir, PROJECT_ROOT, label_vocab, bert_vocab, test_df)
    predictions["behrt"] = behrt_y_prob
    thresholds["behrt"] = REFERENCE_THRESHOLD
    print(f"  done ({time.time() - t0:.1f}s)", flush=True)

    # --- Bootstrap each model on the (UNK-excluded) label subset ---
    print(f"\nBootstrapping ({N_BOOT} resamples per model)...", flush=True)
    summaries = {}
    boot_arrays = {}
    for name, y_prob in predictions.items():
        yt = test_y_true[:, keep_cols]
        yp = y_prob[:, keep_cols]
        summary, boots = summarize(name, yt, yp, thresholds[name], boot_indices)
        summaries[name] = summary
        boot_arrays[name] = boots

    # --- Paired deltas vs BEHRT ---
    print("\nPaired deltas vs BEHRT (95% CI, bootstrap p-value):", flush=True)
    comparisons = []
    point_estimates = {
        name: {
            "aps": summaries[name]["sample_wise_aps"]["point"],
            "auc": summaries[name]["sample_wise_auc"]["point"],
            "f1": summaries[name]["micro_f1"]["point"],
        }
        for name in predictions
    }
    for name in predictions:
        if name == "behrt":
            continue
        for metric_key in ("aps", "auc", "f1"):
            result = paired_delta(
                "behrt", name, boot_arrays["behrt"], boot_arrays[name],
                point_estimates["behrt"], point_estimates[name], metric_key,
            )
            comparisons.append(result)
            sig = "significant" if result["significant_at_0.05"] else "not significant"
            print(f"  behrt - {name:<20} [{metric_key}] delta={result['delta']:+.4f} "
                  f"CI95={result['ci95']} p={result['bootstrap_p_value']:.3f} ({sig})", flush=True)

    output = {
        "n_bootstrap": N_BOOT,
        "rng_seed": RNG_SEED,
        "metric_exclude_labels": ["UNK"],
        "per_model_metrics": summaries,
        "paired_deltas_vs_behrt": comparisons,
    }
    out_path = results_dir / "bootstrap_confidence_intervals.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    print(f"Total time: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
