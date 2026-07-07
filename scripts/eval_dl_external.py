"""External validation of a frozen DL checkpoint on a second dataset (e.g. HMC).

Loads a checkpoint saved by ``run_dl.py`` (architecture + training normalizer +
fitted temperature), evaluates it on an external processed ``.npz`` WITHOUT any
refitting, and writes the results in the SAME shape as the internal test outputs
(a ``test_metrics`` block + probability metrics + a probs ``.npz``). That way
``make_figures.py`` renders the full per-model figure set (confusion matrix,
per-class F1, ROC, PR, reliability) on the external data with no changes.

Leakage-safe by construction: the normalizer and the temperature both come from the
development data only (they are read from the checkpoint, never refit on HMC). This
is the deep-learning counterpart of ``run_external.py`` and avoids retraining by
reusing the checkpoints ``run_dl.py`` already saves.

Usage:
    python scripts/eval_dl_external.py --checkpoint results/logs/dl_cnn_lstm_ctx15.pt \
        --external data/processed/hmc.npz \
        --out results/tables/dl_cnn_lstm_ctx15_hmc.json \
        --probs-out results/logs/dl_cnn_lstm_ctx15_hmc_probs.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data_loader import build_sequence_windows, load_processed_dataset
from src.evaluate import compute_metrics, probabilistic_metrics
from src.models_dl import load_dl_checkpoint
from src.train import (
    MemmapEpochDataset,
    make_lazy_dataloader,
    predict_logits_dl,
    softmax_with_temperature,
)
from src.utils import STAGE_NAMES, ensure_dir, get_logger, save_json, set_seed

logger = get_logger("eval_dl_external")
LABELS = list(range(len(STAGE_NAMES)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frozen DL checkpoint on an external dataset.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint saved by run_dl.py (.pt).")
    parser.add_argument("--external", required=True, help="External processed .npz (x, y, subjects).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default="results/tables/dl_external.json")
    parser.add_argument("--probs-out", default="results/logs/dl_external_probs.npz")
    args = parser.parse_args()

    set_seed()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # The checkpoint carries the architecture, the training normalizer (mean/std) and
    # the temperature fitted on the internal validation set — all reused as-is.
    model, checkpoint = load_dl_checkpoint(args.checkpoint, device=device)
    context = checkpoint["context"]
    mean, std = checkpoint["mean"], checkpoint["std"]
    temperature = checkpoint["temperature"]

    dataset = load_processed_dataset(args.external)
    if dataset.x.shape[1] != checkpoint["n_channels"]:
        raise ValueError(
            f"External data has {dataset.x.shape[1]} channels but the checkpoint expects "
            f"{checkpoint['n_channels']}. Prepare HMC with the same channel layout as Sleep-EDF.")
    logger.info("External set: %d epochs from %d subjects (context=%d).",
                len(dataset.y), dataset.n_subjects, context)

    # Same windowing as the internal test: non-overlapping sequences for context>1.
    idx = np.arange(len(dataset.y))
    item_index = build_sequence_windows(dataset.subjects, idx, context) if context > 1 else idx
    ext_ds = MemmapEpochDataset(dataset.x, dataset.y, item_index, mean, std, context=context)
    ext_loader = make_lazy_dataloader(ext_ds, args.batch_size, num_workers=args.num_workers,
                                      pin_memory=(device == "cuda"))

    y_true, logits = predict_logits_dl(model, ext_loader, device)
    prob_raw = softmax_with_temperature(logits, 1.0)
    prob_cal = softmax_with_temperature(logits, temperature)

    # Named "test_metrics" so make_figures.py treats the external outputs uniformly.
    metrics = compute_metrics(y_true, prob_cal.argmax(axis=1), labels=LABELS)
    metrics_raw = probabilistic_metrics(y_true, prob_raw)
    metrics_cal = probabilistic_metrics(y_true, prob_cal)
    logger.info("External macro-F1: %.3f | kappa: %.3f | ECE cal: %.3f",
                metrics["macro_f1"], metrics["cohen_kappa"], metrics_cal["ece"])

    output = {
        "model": checkpoint["model"],
        "context": context,
        "modalities": checkpoint["modalities"],
        "external_data": args.external,
        "temperature": temperature,
        "test_metrics": metrics,
        "test_prob_metrics_raw": metrics_raw,
        "test_prob_metrics_calibrated": metrics_cal,
    }

    ensure_dir(Path(args.probs_out).parent)
    np.savez_compressed(args.probs_out, y_true=y_true, y_prob=prob_cal, y_prob_raw=prob_raw)
    save_json(output, args.out)
    logger.info("Saved external metrics to %s and probs to %s", args.out, args.probs_out)


if __name__ == "__main__":
    main()
