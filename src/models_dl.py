"""Deep learning models (master plan: Model families -> DL, Advanced model).

Baselines: a 1D-CNN and a CNN-LSTM over raw multi-channel epochs.
Advanced model: a Cross-Modal Transformer that fuses the modalities (EEG / EOG /
EMG) with attention, supporting modality masking for ablations.

Every per-epoch model exposes `encode(x) -> (batch, embed_dim)`; the
SequenceSleepStager reuses that encoder and adds a Transformer across neighboring
epochs (temporal context), predicting one label per epoch. A focal loss is
provided for class imbalance.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import MODALITY_CHANNELS, get_logger

logger = get_logger("models_dl")


class TinyCNN1D(nn.Module):
    """Simple 1D-CNN over raw epochs: the smallest deep baseline."""

    def __init__(self, n_channels, n_classes=5):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = 32
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, n_classes)

    def encode(self, x):
        return self.features(x).squeeze(-1)

    def forward(self, x):
        return self.classifier(self.encode(x))


class CNNLSTM(nn.Module):
    """CNN feature extractor followed by a BiLSTM over the intra-epoch time axis."""

    def __init__(self, n_channels, n_classes=5, hidden=64):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = 2 * hidden
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.lstm = nn.LSTM(32, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, n_classes)

    def encode(self, x):
        x = self.cnn(x).transpose(1, 2)   # (batch, time, channels)
        output, _ = self.lstm(x)
        return output[:, -1, :]

    def forward(self, x):
        return self.classifier(self.encode(x))


class ModalityEncoder(nn.Module):
    """Lightweight 1D-CNN that encodes one modality into ``n_tokens`` temporal tokens.

    Keeps ``n_tokens`` time steps (``AdaptiveAvgPool1d(n_tokens)``) instead of
    collapsing the whole 30 s epoch to a single value. This preserves the transient,
    stage-discriminative events (sleep spindles, K-complexes, eye movements) that a
    global average would wash out, so the cross-modal attention can locate *when*
    within the epoch each modality is informative. Returns (batch, n_tokens, d_model).
    """

    def __init__(self, in_channels, d_model, n_tokens=4):
        super().__init__()
        self.n_tokens = n_tokens
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_tokens),
        )
        self.project = nn.Linear(32, d_model)

    def forward(self, x):
        features = self.cnn(x).transpose(1, 2)   # (batch, n_tokens, 32)
        return self.project(features)            # (batch, n_tokens, d_model)


class CrossModalTransformer(nn.Module):
    """Cross-Modal Transformer for multimodal sleep staging (master plan: Advanced model).

    One encoder per modality -> shared latent space + modality embeddings ->
    transformer fusion across modalities (with optional modality masking) -> a
    CLS token feeding a normalization + dropout + MLP head. `encode` returns the
    fused CLS embedding so the sequence model can reuse it.
    """

    def __init__(self, modality_channels=None, n_classes=5, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.2, active_modalities=None, n_tokens=4):
        super().__init__()
        self.n_classes = n_classes
        self.embed_dim = d_model
        self.n_tokens = n_tokens
        self.modality_channels = modality_channels or MODALITY_CHANNELS
        self.modality_names = list(self.modality_channels)
        self.active_modalities = active_modalities or list(self.modality_names)

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(len(channels), d_model, n_tokens=n_tokens)
            for name, channels in self.modality_channels.items()
        })
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # One embedding per token slot: index 0 = CLS, 1..n = the modalities.
        self.modality_embedding = nn.Parameter(torch.zeros(1, len(self.modality_names) + 1, d_model))
        # Within-epoch position of each modality token (the n_tokens are ordered in time,
        # so attention needs a positional signal to tell early from late in the epoch).
        self.token_positional = nn.Parameter(torch.zeros(1, n_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_classes),
        )

    def encode(self, x, active_modalities=None):
        """Return the fused CLS embedding (batch, d_model).

        Each modality contributes ``n_tokens`` temporal tokens (not one), tagged with a
        modality embedding and a within-epoch positional embedding; a CLS token gathers
        the fused representation. Inactive modalities are masked out (all their tokens),
        which is what the ablation / missing-modality experiments rely on.
        """
        active = active_modalities or self.active_modalities
        batch = x.shape[0]
        device = x.device

        cls = self.cls_token.expand(batch, -1, -1) + self.modality_embedding[:, :1, :]
        tokens = [cls]
        padding = [torch.zeros(batch, 1, dtype=torch.bool, device=device)]
        for i, name in enumerate(self.modality_names):
            channels = self.modality_channels[name]
            token_group = self.encoders[name](x[:, channels, :])  # (batch, n_tokens, d_model)
            token_group = token_group + self.modality_embedding[:, i + 1:i + 2, :] + self.token_positional
            tokens.append(token_group)
            padding.append(torch.full((batch, token_group.shape[1]), name not in active,
                                      dtype=torch.bool, device=device))

        tokens = torch.cat(tokens, dim=1)
        key_padding_mask = torch.cat(padding, dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        return encoded[:, 0]

    def forward(self, x, active_modalities=None):
        return self.classifier(self.encode(x, active_modalities))


class SequenceSleepStager(nn.Module):
    """Hierarchical model adding temporal context across neighboring epochs.

    A per-epoch encoder produces one embedding per epoch; a Transformer with
    positional encoding then attends across the epoch sequence and predicts a
    label for every epoch (many-to-many). This mirrors how AASM scoring and SOTA
    sequence models (DeepSleepNet / SeqSleepNet) use inter-epoch context.

    Input:  (batch, seq_len, n_channels, n_samples)
    Output: (batch, seq_len, n_classes)
    """

    def __init__(self, epoch_encoder, n_classes=5, n_heads=4, n_layers=2, max_len=128, dropout=0.2):
        super().__init__()
        self.n_classes = n_classes
        self.epoch_encoder = epoch_encoder
        d_model = epoch_encoder.embed_dim
        self.positional_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True
        )
        self.sequence_model = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        batch, seq_len = x.shape[0], x.shape[1]
        flat = x.reshape(batch * seq_len, *x.shape[2:])
        embeddings = self.epoch_encoder.encode(flat).reshape(batch, seq_len, -1)
        embeddings = embeddings + self.positional_embedding[:, :seq_len]
        encoded = self.sequence_model(embeddings)
        return self.classifier(encoded)


class FocalLoss(nn.Module):
    """Multi-class focal loss for imbalanced sleep staging (especially N1)."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        cross_entropy = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        probability = torch.exp(-cross_entropy)
        return ((1 - probability) ** self.gamma * cross_entropy).mean()


def build_dl_model(name, n_channels, n_classes=5, active_modalities=None):
    """Return an untrained per-epoch model by name: 'cnn', 'cnn_lstm' or 'transformer'."""
    name = name.lower()
    if name == "cnn":
        return TinyCNN1D(n_channels, n_classes)
    if name == "cnn_lstm":
        return CNNLSTM(n_channels, n_classes)
    if name == "transformer":
        return CrossModalTransformer(n_classes=n_classes, active_modalities=active_modalities)
    raise ValueError(f"Unknown DL model: {name}")


def build_sequence_model(name, n_channels, n_classes=5, active_modalities=None, max_len=128):
    """Wrap a per-epoch encoder in a SequenceSleepStager for temporal context."""
    epoch_encoder = build_dl_model(name, n_channels, n_classes, active_modalities=active_modalities)
    return SequenceSleepStager(epoch_encoder, n_classes=n_classes, max_len=max_len)


def load_dl_checkpoint(path, device="cpu"):
    """Rebuild a trained DL model from a checkpoint saved by ``run_dl.py``.

    The checkpoint carries the architecture config (model name, context, channels,
    modalities) plus the training normalizer and temperature, so the exact model can
    be reconstructed for evaluation or explainability (e.g. Grad-CAM). Returns
    (model, checkpoint_dict); the model is on ``device`` in eval mode.

    ``weights_only=False`` because the checkpoint also stores the numpy normalizer.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["context"] > 1:
        model = build_sequence_model(checkpoint["model"], checkpoint["n_channels"],
                                     checkpoint["n_classes"], active_modalities=checkpoint["modalities"],
                                     max_len=checkpoint["max_len"])
    else:
        model = build_dl_model(checkpoint["model"], checkpoint["n_channels"],
                               checkpoint["n_classes"], active_modalities=checkpoint["modalities"])
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    # Checkpoints saved before the additive `token_positional` embedding lack that
    # key. It is zero-initialized and purely additive, so leaving it at zeros
    # reproduces the trained model exactly; any other gap is a genuine mismatch.
    only_token_positional = all(key.endswith("token_positional") for key in incompatible.missing_keys)
    if incompatible.unexpected_keys or not only_token_positional:
        raise RuntimeError(
            f"Incompatible checkpoint {path}: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}.")
    if incompatible.missing_keys:
        logger.warning("Checkpoint %s predates 'token_positional'; using zeros "
                       "(reproduces the trained model).", Path(path).name)
    model.to(device).eval()
    return model, checkpoint
