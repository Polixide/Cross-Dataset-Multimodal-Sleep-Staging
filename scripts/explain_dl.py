"""Grad-CAM explainability figures for a trained per-epoch DL model.

Loads a context-1 checkpoint saved by ``run_dl.py`` (the per-epoch classifier),
finds a correctly classified and a misclassified example for each sleep stage,
and saves a stacked multi-channel overlay per example: every channel's signal
drawn over its Grad-CAM saliency on a shared time axis (master plan: Grad-CAM for
DL, "correct vs misclassified" explainability figures).

Two architectures are handled with the method that actually fits them:
- ``cnn``         -> one Grad-CAM over the last convolution + a per-channel
                     importance bar (gradient x input).
- ``transformer`` -> a *per-modality* Grad-CAM (EEG/EOG/EMG, one map each) + a
                     modality-importance bar taken from the CLS cross-attention
                     when the PyTorch build exposes it, else from Grad-CAM mass.

Works on any processed dataset (Sleep-EDF, HMC, ISRUC) — pass --channel-names to
match a non-Sleep-EDF montage. Grad-CAM is epoch-level, so use a context-1
checkpoint (e.g. dl_transformer_ctx1.pt); a sequence checkpoint is rejected with a
hint, because its per-epoch head is not the one used for scoring.

Usage:
    python scripts/explain_dl.py --checkpoint results/logs/dl_transformer_ctx1.pt \
        --data data/processed/sleep_edf.npz
    python scripts/explain_dl.py --checkpoint results/logs/dl_cnn_ctx1.pt \
        --data data/processed/isruc.npz --channel-names C3-A2 C4-A1 LOC-A2 X1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.data_loader import load_processed_dataset, subject_wise_split
from src.explainability import (
    channel_saliency,
    grad_cam_1d,
    grad_cam_per_modality,
    modality_importance_from_cams,
    plot_epoch_explanation,
    transformer_cls_attention,
    _is_cross_modal,
)
from src.models_dl import load_dl_checkpoint
from src.train import softmax_with_temperature
from src.utils import SLEEP_EDF_CHANNELS, STAGE_NAMES, get_logger, set_seed

logger = get_logger("explain_dl")


def predict_subset(model, x, y, idx, mean, std, temperature, batch_size=128):
    """Return (pred, prob) for the epochs in ``idx`` using training normalization."""
    preds, probs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            raw = np.asarray(x[batch_idx], dtype=np.float32)
            norm = (raw - mean[None]) / std[None]            # (b, C, T)
            logits = model(torch.from_numpy(norm)).numpy()
            prob = softmax_with_temperature(logits, temperature)
            probs.append(prob)
            preds.append(prob.argmax(axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def normalized_epoch(x, gidx, mean, std):
    """Read one raw epoch from the (memmapped) array and z-score it as the model saw it."""
    raw = np.asarray(x[gidx], dtype=np.float32)
    return (raw - mean) / std


def explain_example(model, sample, pred_class, is_transformer, channel_names):
    """Return (cams, importance, importance_label) for one epoch and its prediction."""
    if is_transformer:
        cams = grad_cam_per_modality(model, sample, pred_class)
        attention = transformer_cls_attention(model, sample)
        if attention is not None:
            names = {"eeg": "EEG", "eog": "EOG", "emg": "EMG"}
            importance = {names.get(k, k.upper()): v for k, v in attention["modality_share"].items()}
            return cams, importance, "CLS attention share"
        share = modality_importance_from_cams(cams)
        importance = {k.upper(): v for k, v in share.items()}
        return cams, importance, "Grad-CAM mass share"

    cam = grad_cam_1d(model, sample, pred_class)
    saliency = channel_saliency(model, sample, pred_class)
    importance = {name: float(value) for name, value in zip(channel_names, saliency)}
    return cam, importance, "Channel |grad x input|"


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM figures for a per-epoch DL model.")
    parser.add_argument("--checkpoint", required=True, help="Context-1 checkpoint from run_dl.py.")
    parser.add_argument("--data", required=True, help="Processed .npz (raw epochs).")
    parser.add_argument("--out-dir", default="results/figures/xai")
    parser.add_argument("--classes", nargs="+", type=int, default=list(range(len(STAGE_NAMES))),
                        help="Stage indices to explain (default: all 5).")
    parser.add_argument("--channel-names", nargs="+", default=SLEEP_EDF_CHANNELS,
                        help="Channel labels for the figure (match the dataset montage).")
    parser.add_argument("--max-scan", type=int, default=4000,
                        help="How many epochs to scan when picking example epochs.")
    parser.add_argument("--test-only", action="store_true",
                        help="Pick examples only from the held-out subject-wise test split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Grad-CAM runs on CPU: the per-epoch models are tiny and it keeps the backward
    # hooks and numpy interop simple.
    model, checkpoint = load_dl_checkpoint(args.checkpoint, device="cpu")
    if checkpoint["context"] > 1:
        raise SystemExit(
            f"{Path(args.checkpoint).name} is a sequence checkpoint (context="
            f"{checkpoint['context']}). Grad-CAM here is epoch-level — use a context-1 "
            "checkpoint (e.g. dl_transformer_ctx1.pt / dl_cnn_ctx1.pt).")

    is_transformer = _is_cross_modal(model)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(-1, 1)   # (C, 1)
    std = np.asarray(checkpoint["std"], dtype=np.float32).reshape(-1, 1)
    temperature = checkpoint["temperature"]
    sfreq = checkpoint.get("sfreq") or load_processed_dataset(args.data).sfreq or 100.0

    dataset = load_processed_dataset(args.data)
    if dataset.x.shape[1] != len(args.channel_names):
        raise ValueError(f"{len(args.channel_names)} channel names but data has "
                         f"{dataset.x.shape[1]} channels.")
    y = dataset.y

    # Scan a subset to find confident correct / error examples per class. Restrict
    # to the held-out test subjects when asked, so the explanation is on unseen data.
    pool = np.arange(len(y))
    if args.test_only:
        _, _, pool = subject_wise_split(dataset.subjects)
    rng = np.random.default_rng(args.seed)
    scan = np.sort(rng.choice(pool, size=min(args.max_scan, len(pool)), replace=False))
    logger.info("Scanning %d epochs to pick examples (%s)...", len(scan), checkpoint["model"])
    pred, prob = predict_subset(model, dataset.x, y, scan, mean, std, temperature)
    confidence = prob.max(axis=1)

    out_dir = Path(args.out_dir) / f"{checkpoint['model']}_ctx1"
    saved = []
    for c in args.classes:
        true_c = y[scan] == c
        for kind, mask in (("correct", true_c & (pred == c)), ("error", true_c & (pred != c))):
            if not mask.any():
                logger.warning("No %s example for stage %s in the scanned subset.", kind, STAGE_NAMES[c])
                continue
            local = np.where(mask)[0]
            chosen = local[np.argmax(confidence[local])]      # most confident such epoch
            gidx = int(scan[chosen])
            pred_class = int(pred[chosen])

            sample = normalized_epoch(dataset.x, gidx, mean, std)
            cams, importance, importance_label = explain_example(
                model, sample, pred_class, is_transformer, args.channel_names)
            title = (f"{checkpoint['model']} | true={STAGE_NAMES[c]} pred={STAGE_NAMES[pred_class]} "
                     f"p={confidence[chosen]:.2f} | {kind}")
            out_path = out_dir / f"{STAGE_NAMES[c]}_{kind}.png"
            plot_epoch_explanation(sample, cams, args.channel_names, out_path, title=title,
                                   sfreq=sfreq, importance=importance,
                                   importance_label=importance_label or "Importance")
            saved.append(out_path)
            logger.info("Saved %s", out_path)

    logger.info("Done: %d Grad-CAM figures under %s", len(saved), out_dir)


if __name__ == "__main__":
    main()
