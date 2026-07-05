"""Train and evaluate a deep-learning model on raw epochs (subject-wise).

Builds a 1D-CNN, CNN-LSTM or Cross-Modal Transformer, normalizes with
training-only statistics, trains with class-imbalance handling, calibrates with
temperature scaling, and evaluates on the held-out internal test subjects.

With --context N > 1 the per-epoch model becomes the epoch encoder of a
hierarchical sequence model (SequenceSleepStager): sequences of N consecutive
epochs are built within each subject and a Transformer attends across them,
predicting one label per epoch (temporal context).

Optionally: random-search tuning, modality ablations, missing-modality
robustness, external validation, class-balanced mini-batches.

Examples:
    python scripts/run_dl.py --data data/processed/sleep_edf.npz --model transformer
    python scripts/run_dl.py --data data/processed/sleep_edf.npz --model cnn --context 15
    python scripts/run_dl.py --data data/processed/sleep_edf.npz --model transformer \
        --context 11 --external data/processed/hmc.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data_loader import build_sequence_windows, load_processed_dataset, subject_wise_split
from src.evaluate import compute_metrics, probabilistic_metrics
from src.models_dl import build_dl_model, build_sequence_model
from src.preprocessing import fit_normalizer_streaming
from src.train import (
    MemmapEpochDataset,
    compute_class_weights,
    evaluate_dl,
    fit_temperature,
    make_lazy_dataloader,
    predict_logits_dl,
    random_search_dl,
    softmax_with_temperature,
    train_dl,
)
from src.utils import RANDOM_SEED, STAGE_NAMES, ensure_dir, get_logger, save_json, set_seed

logger = get_logger("run_dl")
LABELS = list(range(len(STAGE_NAMES)))


def evaluate_missing_modality(model, test_loader):
    """Evaluate the trained transformer with each modality dropped in turn."""
    baseline = model.active_modalities
    results = {}
    for dropped in model.modality_names:
        model.active_modalities = [name for name in model.modality_names if name != dropped]
        results[f"drop_{dropped}"] = evaluate_dl(model, test_loader)["macro_f1"]
    model.active_modalities = baseline
    return results


def build_split_dataset(x, y, subjects, split_idx, mean, std, context):
    """Build a lazy, memmap-backed dataset for one subject-wise split.

    Signals are read on the fly from the memmap and normalized per item, so the
    full array never enters RAM. With context > 1 the item indices are sequences
    of consecutive within-subject epochs (temporal context).
    """
    if context > 1:
        item_index = build_sequence_windows(subjects, split_idx, context)
    else:
        item_index = np.asarray(split_idx)
    return MemmapEpochDataset(x, y, item_index, mean, std, context=context)


def main():
    parser = argparse.ArgumentParser(description="Run a deep-learning sleep-staging model.")
    parser.add_argument("--data", required=True, help="Processed .npz with (x, y, subjects).")
    parser.add_argument("--model", default="cnn", choices=["cnn", "cnn_lstm", "transformer"])
    parser.add_argument("--context", type=int, default=1,
                        help="Epochs per sequence (>1 enables temporal context).")
    parser.add_argument("--modalities", nargs="+", default=["eeg", "eog", "emg"],
                        choices=["eeg", "eog", "emg"], help="Active modalities (transformer only).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--loss", default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--tune", action="store_true", help="Random-search learning rate and loss.")
    parser.add_argument("--balanced-batches", action="store_true",
                        help="Draw class-balanced mini-batches (per-epoch only).")
    parser.add_argument("--external", help="Optional processed .npz for external validation.")
    parser.add_argument("--missing-modality-test", action="store_true",
                        help="Evaluate the transformer with each modality dropped (context=1).")
    parser.add_argument("--out", default="results/tables/dl_metrics.json")
    parser.add_argument("--probs-out", default="results/logs/dl_test_probs.npz")
    args = parser.parse_args()

    set_seed()
    torch.manual_seed(RANDOM_SEED)

    dataset = load_processed_dataset(args.data)
    subjects = dataset.subjects
    train_idx, val_idx, test_idx = subject_wise_split(subjects)
    logger.info("Subject-wise split -> train=%d val=%d test=%d epochs (context=%d).",
                len(train_idx), len(val_idx), len(test_idx), args.context)

    # Per-channel statistics from the training subjects only, streamed off the
    # memmap so the full (tens-of-GB) signal array is never materialized in RAM.
    mean, std = fit_normalizer_streaming(dataset.x, train_idx)

    train_ds = build_split_dataset(dataset.x, dataset.y, subjects, train_idx, mean, std, args.context)
    val_ds = build_split_dataset(dataset.x, dataset.y, subjects, val_idx, mean, std, args.context)
    test_ds = build_split_dataset(dataset.x, dataset.y, subjects, test_idx, mean, std, args.context)

    train_loader = make_lazy_dataloader(train_ds, args.batch_size,
                                        shuffle=True, balanced=args.balanced_batches)
    val_loader = make_lazy_dataloader(val_ds, args.batch_size)
    test_loader = make_lazy_dataloader(test_ds, args.batch_size)

    n_channels = dataset.x.shape[1]
    class_weights = compute_class_weights(train_ds.labels)

    def model_factory():
        if args.context > 1:
            return build_sequence_model(args.model, n_channels, len(STAGE_NAMES),
                                        active_modalities=args.modalities, max_len=max(args.context, 8))
        return build_dl_model(args.model, n_channels, len(STAGE_NAMES), active_modalities=args.modalities)

    output = {"model": args.model, "context": args.context, "modalities": args.modalities, "loss": args.loss}
    if args.tune:
        best, trials = random_search_dl(model_factory, train_loader, val_loader,
                                        epochs=args.epochs, class_weights=class_weights)
        model = best["model"]
        output["tuning"] = {"best": {k: best[k] for k in ("lr", "use_focal", "val_macro_f1")}, "trials": trials}
        logger.info("Best DL hyperparameters: lr=%s focal=%s (val macro-F1 %.3f)",
                    best["lr"], best["use_focal"], best["val_macro_f1"])
    else:
        model, history = train_dl(model_factory(), train_loader, val_loader, epochs=args.epochs,
                                  class_weights=class_weights, use_focal=(args.loss == "focal"))
        output["history"] = history

    # Temperature scaling on validation, then evaluate on the held-out test.
    y_val_true, val_logits = predict_logits_dl(model, val_loader)
    y_test_true, test_logits = predict_logits_dl(model, test_loader)
    temperature = fit_temperature(val_logits, y_val_true)
    prob_raw = softmax_with_temperature(test_logits, 1.0)
    prob_cal = softmax_with_temperature(test_logits, temperature)

    test_metrics = compute_metrics(y_test_true, prob_raw.argmax(axis=1), labels=LABELS)
    metrics_raw = probabilistic_metrics(y_test_true, prob_raw)
    metrics_cal = probabilistic_metrics(y_test_true, prob_cal)
    logger.info("Test macro-F1: %.3f | ECE raw %.3f -> calibrated %.3f (T=%.2f)",
                test_metrics["macro_f1"], metrics_raw["ece"], metrics_cal["ece"], temperature)

    output.update({
        "temperature": temperature,
        "test_metrics": test_metrics,
        "test_prob_metrics_raw": metrics_raw,
        "test_prob_metrics_calibrated": metrics_cal,
    })

    if args.missing_modality_test and args.model == "transformer" and args.context == 1:
        output["missing_modality_macro_f1"] = evaluate_missing_modality(model, test_loader)
        logger.info("Missing-modality macro-F1: %s", output["missing_modality_macro_f1"])

    if args.external:
        external = load_processed_dataset(args.external)
        # Reuse the training normalizer (no refit) and read the external memmap lazily.
        external_idx = np.arange(len(external.y))
        external_ds = build_split_dataset(external.x, external.y, external.subjects,
                                          external_idx, mean, std, args.context)
        external_loader = make_lazy_dataloader(external_ds, args.batch_size)
        output["external_metrics"] = evaluate_dl(model, external_loader)
        logger.info("External macro-F1: %.3f", output["external_metrics"]["macro_f1"])

    ensure_dir(Path(args.probs_out).parent)
    np.savez_compressed(args.probs_out, y_true=y_test_true, y_prob=prob_cal, y_prob_raw=prob_raw)
    save_json(output, args.out)
    logger.info("Saved DL metrics to %s", args.out)


if __name__ == "__main__":
    main()
