"""General utilities: reproducibility, logging, IO, and project-wide constants.

This module centralizes small helpers and the canonical label/channel
definitions, so every stage of the pipeline uses the same 5-class convention.
"""
import json
import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np

# 5-class AASM sleep staging (master plan: Labels and class definition).
STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_TO_INDEX = {name: index for index, name in enumerate(STAGE_NAMES)}
INDEX_TO_STAGE = {index: name for name, index in STAGE_TO_INDEX.items()}

# Epoch length used for scoring (master plan: Methods -> Epoching).
EPOCH_SECONDS = 30

# Sleep-EDF channels of interest (master plan: Data -> Sleep-EDF Expanded).
SLEEP_EDF_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]

# Which channel indices (into SLEEP_EDF_CHANNELS) belong to each modality.
# Used by the cross-modal Transformer to build one encoder per modality.
MODALITY_CHANNELS = {"eeg": [0, 1], "eog": [2], "emg": [3]}

RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """Seed Python and NumPy for reproducible runs.

    Deep-learning scripts seed torch themselves (torch.manual_seed) so this
    module stays lightweight and free of heavy imports.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name="sleep_staging", level=logging.INFO):
    """Return a configured logger with a single stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path):
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path):
    """Write an object to JSON, creating parent folders if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, default=str)


def load_json(path):
    """Read an object from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(obj, path):
    """Write an object to a pickle file, creating parent folders if needed."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    """Read an object from a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)
