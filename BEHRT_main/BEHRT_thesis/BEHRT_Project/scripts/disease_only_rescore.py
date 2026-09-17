#!/usr/bin/env python3
"""
Diagnostic (NO retraining): re-scores the ALREADY-TRAINED reference BEHRT
checkpoint on the test set, restricted to disease-only CCSR labels, to check
whether excluding symptom / administrative / injury / pregnancy-and-perinatal
codes from the prediction target actually improves aggregate ranking metrics
-- before committing to a full retrain with a disease-only label vocabulary
and output layer.

This reuses the exact same label vocabulary, data splits, and eval functions
as scripts/train_nextvisit_clean.py, but only runs a forward pass (inference)
over the existing best checkpoint -- no backprop, no training loop.

Usage:
    RUN_DIR=data/models/clean_runs/clean_run_20260910_124255 \
        python3 scripts/disease_only_rescore.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.common import load_obj
from model.utils import age_vocab
from dataLoader.NextXVisit import NextVisit
from model.NextXVisit import BertForMultiLabelPrediction
from scripts.train_nextvisit_clean import (
    BertConfig,
    build_label_subset,
    collect_eval_arrays,
    compute_new_recurring_metrics,
    compute_recurring_mask,
    evaluate_from_arrays,
    format_label_vocab,
    tune_decision_threshold,
)

# Same category -> code-type mapping used in eda/11_code_type_performance_breakdown.py
# (including the known MAL mislabeling, kept as-is per prior decision not to fix it now;
# it affects only 9 labels / 417 test occurrences and does not change this diagnostic's
# conclusion either way).
CATEGORY_TO_CODE_TYPE = {
    "SYM": "symptom",
    "FAC": "administrative",
    "UTL": "administrative",
    "INJ": "injury",
    "PRG": "pregnancy_and_perinatal",
    "PNL": "pregnancy_and_perinatal",
    "CON": "pregnancy_and_perinatal",
    "MAL": "administrative",
}


def label_code_type(label):
    # labels look like "CCSR_<CAT><NUM>"
    if not label.startswith("CCSR_"):
        return "other"
    rest = label[len("CCSR_"):]
    cat = "".join(ch for ch in rest if ch.isalpha())
    return CATEGORY_TO_CODE_TYPE.get(cat, "disease")


def main():
    import pandas as pd

    data_dir = PROJECT_ROOT / "data" / "processed"
    run_dir = PROJECT_ROOT / "data" / "models" / "clean_runs" / os.getenv(
        "RUN_ID", "clean_run_20260910_124255"
    )
    checkpoint_path = run_dir / "behrt_nextvisit_ccsr_clean_best.pt"

    print(f"Loading checkpoint from {checkpoint_path}", flush=True)

    train_df = pd.read_parquet(data_dir / "train_nextvisit_ccsr_clean.parquet")
    val_df = pd.read_parquet(data_dir / "val_nextvisit_ccsr_clean.parquet")
    test_df = pd.read_parquet(data_dir / "test_nextvisit_ccsr_clean.parquet")
    bert_vocab = load_obj(str(data_dir / "vocab_ccsr_clean"))
    age_vocab_dict, _ = age_vocab(max_age=110, symbol=None)

    base_label_vocab = format_label_vocab(bert_vocab["token2idx"])
    label_vocab, train_df, _ = build_label_subset(train_df, base_label_vocab, top_k_labels=0, min_label_freq=0.0)
    print(f"Label vocab size: {len(label_vocab)}", flush=True)

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
    print("Checkpoint loaded.", flush=True)

    val_set = NextVisit(
        token2idx=bert_vocab["token2idx"], label2idx=label_vocab, age2idx=age_vocab_dict,
        dataframe=val_df, max_len=global_params["max_len_seq"],
    )
    test_set = NextVisit(
        token2idx=bert_vocab["token2idx"], label2idx=label_vocab, age2idx=age_vocab_dict,
        dataframe=test_df, max_len=global_params["max_len_seq"],
    )
    val_loader = DataLoader(val_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=global_params["batch_size"], shuffle=False, num_workers=0)

    print("Running inference on val/test (no training)...", flush=True)
    val_y_true, val_y_prob = collect_eval_arrays(model, val_loader, mlb, global_params["device"])
    test_y_true, test_y_prob = collect_eval_arrays(model, test_loader, mlb, global_params["device"])
    print(f"  val shape={val_y_prob.shape} test shape={test_y_prob.shape}", flush=True)

    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    code_types = {idx: label_code_type(label) for idx, label in idx_to_label.items()}
    non_disease_labels = [idx_to_label[idx] for idx, ct in code_types.items() if ct != "disease"]
    non_disease_labels.append("UNK")
    print(f"Excluding {len(non_disease_labels)} non-disease/UNK labels out of {len(label_vocab)}", flush=True)

    # --- Full vocabulary (current paper numbers), for a sanity-check baseline ---
    full_threshold, _, _ = tune_decision_threshold(val_y_true, val_y_prob, metric_name="micro_f1")
    full_metrics = evaluate_from_arrays(
        test_y_true, test_y_prob, threshold=full_threshold,
        label_vocab=label_vocab, train_supports={}, include_per_class=False,
        exclude_labels=("UNK",),
    )
    print(f"\n[Full vocab, UNK excluded] threshold={full_threshold:.2f} "
          f"APS={full_metrics['sample_wise_aps']:.4f} AUC={full_metrics['sample_wise_auc']:.4f} "
          f"micro_f1={full_metrics['micro_f1']:.4f}", flush=True)

    # --- Disease-only restricted vocabulary ---
    disease_threshold, _, _ = tune_decision_threshold(
        val_y_true[:, [i for i in range(val_y_true.shape[1]) if idx_to_label[i] not in non_disease_labels]],
        val_y_prob[:, [i for i in range(val_y_prob.shape[1]) if idx_to_label[i] not in non_disease_labels]],
        metric_name="micro_f1",
    )
    disease_metrics = evaluate_from_arrays(
        test_y_true, test_y_prob, threshold=disease_threshold,
        label_vocab=label_vocab, train_supports={}, include_per_class=False,
        exclude_labels=tuple(non_disease_labels),
    )
    print(f"[Disease-only] threshold={disease_threshold:.2f} "
          f"APS={disease_metrics['sample_wise_aps']:.4f} AUC={disease_metrics['sample_wise_auc']:.4f} "
          f"micro_f1={disease_metrics['micro_f1']:.4f}", flush=True)

    print("\n=== Summary (existing checkpoint, re-scored, no retraining) ===")
    print(f"{'Scope':<20}{'APS':>10}{'AUC':>10}{'Micro-F1':>12}{'Threshold':>12}")
    print(f"{'Full vocab':<20}{full_metrics['sample_wise_aps']:>10.4f}"
          f"{full_metrics['sample_wise_auc']:>10.4f}{full_metrics['micro_f1']:>12.4f}"
          f"{full_threshold:>12.2f}")
    print(f"{'Disease-only':<20}{disease_metrics['sample_wise_aps']:>10.4f}"
          f"{disease_metrics['sample_wise_auc']:>10.4f}{disease_metrics['micro_f1']:>12.4f}"
          f"{disease_threshold:>12.2f}")


if __name__ == "__main__":
    main()
