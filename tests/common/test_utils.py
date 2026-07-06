"""Tests for the shared utility helpers (IO for the comparison table)."""
import pandas as pd

from src.common.utils import append_metrics_row

KEY_COLS = ["model", "balance"]


def test_append_creates_and_accumulates_rows(tmp_path):
    path = tmp_path / "ml_metrics.csv"
    append_metrics_row({"model": "logreg", "balance": "class_weight", "macro_f1": 0.70}, path, KEY_COLS)
    append_metrics_row({"model": "rf", "balance": "class_weight", "macro_f1": 0.75}, path, KEY_COLS)

    table = pd.read_csv(path)
    assert len(table) == 2
    assert set(table["model"]) == {"logreg", "rf"}


def test_rerun_same_config_replaces_row(tmp_path):
    path = tmp_path / "ml_metrics.csv"
    append_metrics_row({"model": "logreg", "balance": "class_weight", "macro_f1": 0.70}, path, KEY_COLS)
    append_metrics_row({"model": "logreg", "balance": "class_weight", "macro_f1": 0.80}, path, KEY_COLS)

    table = pd.read_csv(path)
    assert len(table) == 1
    assert table.loc[0, "macro_f1"] == 0.80


def test_different_balance_is_a_new_row(tmp_path):
    path = tmp_path / "ml_metrics.csv"
    append_metrics_row({"model": "logreg", "balance": "class_weight", "macro_f1": 0.70}, path, KEY_COLS)
    append_metrics_row({"model": "logreg", "balance": "smote", "macro_f1": 0.72}, path, KEY_COLS)

    table = pd.read_csv(path)
    assert len(table) == 2
