"""Tests for subject-wise splitting and the leakage guard.

These protect the most important methodological rule of the project: no subject
may appear in more than one split. (Test files use plain `assert`, which is the
standard pytest style.)
"""
import numpy as np
import pytest

from src.common.data import (
    check_no_subject_overlap,
    make_epoch_sequences,
    subject_wise_folds,
    subject_wise_split,
)


def make_subjects(n_subjects=10, epochs_per=20):
    """Build a subject vector: epochs_per consecutive epochs per subject."""
    return np.repeat(np.arange(n_subjects), epochs_per)


def test_subject_wise_split_is_disjoint_by_subject():
    subjects = make_subjects()
    train, val, test = subject_wise_split(subjects, seed=0)

    train_subjects = set(subjects[train])
    val_subjects = set(subjects[val])
    test_subjects = set(subjects[test])
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)
    # every epoch is assigned to exactly one split
    assert len(train) + len(val) + len(test) == len(subjects)


def test_split_is_deterministic_for_a_seed():
    subjects = make_subjects()
    first = subject_wise_split(subjects, seed=123)
    second = subject_wise_split(subjects, seed=123)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_check_no_subject_overlap_raises_on_leak():
    with pytest.raises(ValueError):
        check_no_subject_overlap([1, 2, 3], [3, 4])


def test_check_no_subject_overlap_passes_when_disjoint():
    check_no_subject_overlap([1, 2], [3, 4], [5])  # should not raise


def test_subject_wise_folds_have_no_shared_subject():
    subjects = make_subjects()
    for train_idx, val_idx in subject_wise_folds(subjects, n_splits=5):
        assert set(subjects[train_idx]).isdisjoint(set(subjects[val_idx]))


def test_epoch_sequences_stay_within_subjects():
    x = np.arange(12 * 2).reshape(12, 2, 1).astype(float)  # 12 epochs, 2 channels, 1 sample
    y = np.arange(12)
    subjects = np.array([0] * 6 + [1] * 6)

    x_seq, y_seq, window_subjects = make_epoch_sequences(x, y, subjects, seq_len=3)

    # 6 epochs / 3 -> 2 non-overlapping windows per subject = 4 windows
    assert x_seq.shape == (4, 3, 2, 1)
    assert y_seq.shape == (4, 3)
    assert set(window_subjects.tolist()) == {0, 1}
    # each window holds consecutive epochs (temporal order preserved, no crossing)
    for row in y_seq:
        assert np.array_equal(row, np.arange(row[0], row[0] + 3))
