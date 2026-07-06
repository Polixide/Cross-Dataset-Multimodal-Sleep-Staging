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
    python scripts/dl/run_dl.py --data data/processed/sleep_edf.npz --model transformer
    python scripts/dl/run_dl.py --data data/processed/sleep_edf.npz --model cnn --context 15
    python scripts/dl/run_dl.py --data data/processed/sleep_edf.npz --model transformer \
        --context 11 --external data/processed/shhs.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.common.data import load_processed_dataset, make_epoch_sequences, subject_wise_split
from src.common.evaluation import compute_metrics, probabilistic_metrics
from src.dl.models import build_dl_model, build_sequence_model
from src.common.preprocessing import apply_normalizer, fit_normalizer
from src.dl.train import (
    compute_class_weights,
    evaluate_dl,
    fit_temperature,
    make_dataloader,
    predict_logits_dl,
    random_search_dl,
    softmax_with_temperature,
    train_dl,
)
from src.common.utils import RANDOM_SEED, STAGE_NAMES, ensure_dir, get_logger, save_json, set_seed

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


def split_epochs(x_norm, y, subjects, idx, context):
    """Return the (x, y) for a split, as sequences when context > 1."""
    if context > 1:
        x_seq, y_seq, _ = make_epoch_sequences(x_norm[idx], y[idx], subjects[idx], context)
        return x_seq, y_seq
    return x_norm[idx], y[idx]


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

    # Normalize every epoch with statistics from the training subjects only.
    mean, std = fit_normalizer(dataset.x[train_idx])
    x_norm = apply_normalizer(dataset.x, mean, std)

    x_train, y_train = split_epochs(x_norm, dataset.y, subjects, train_idx, args.context)
    x_val, y_val = split_epochs(x_norm, dataset.y, subjects, val_idx, args.context)
    x_test, y_test = split_epochs(x_norm, dataset.y, subjects, test_idx, args.context)

    train_loader = make_dataloader(x_train, y_train, args.batch_size,
                                   shuffle=True, balanced=args.balanced_batches)
    val_loader = make_dataloader(x_val, y_val, args.batch_size)
    test_loader = make_dataloader(x_test, y_test, args.batch_size)

    n_channels = dataset.x.shape[1]
    class_weights = compute_class_weights(np.asarray(y_train).reshape(-1))

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
        x_external = apply_normalizer(external.x, mean, std)
        xe, ye = (make_epoch_sequences(x_external, external.y, external.subjects, args.context)[:2]
                  if args.context > 1 else (x_external, external.y))
        external_loader = make_dataloader(xe, ye, args.batch_size)
        output["external_metrics"] = evaluate_dl(model, external_loader)
        logger.info("External macro-F1: %.3f", output["external_metrics"]["macro_f1"])

    ensure_dir(Path(args.probs_out).parent)
    np.savez_compressed(args.probs_out, y_true=y_test_true, y_prob=prob_cal, y_prob_raw=prob_raw)
    save_json(output, args.out)
    logger.info("Saved DL metrics to %s", args.out)


if __name__ == "__main__":
    main()
