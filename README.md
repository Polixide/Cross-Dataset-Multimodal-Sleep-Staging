# Cross-Dataset Multimodal Sleep Staging Benchmark

**Subject-Independent Learning, External Validation, Calibration, and Explainable AI.**

A benchmark for automatic 5-class sleep staging (Wake, N1, N2, N3, REM) that is
designed around *robust methodology* rather than a single leaderboard number.
Models are trained on one public PSG dataset and evaluated **on unseen subjects
and on a second, external dataset**, with class-imbalance handling, probability
calibration, and explainability built into the pipeline from the start.

> University project for the *Artificial Intelligence in Medicine* course.

## Why this project

Many published sleep-staging models report strong *within-dataset* accuracy but
rely on epoch-wise random splits (which leak subjects across train/test), skip
external validation, and omit calibration and explainability. This project
closes that gap with a single coherent benchmark:

- **Subject-independent evaluation** — all splits are subject-wise; no subject
  appears in more than one split.
- **External validation** — a model trained on Sleep-EDF is tested, frozen, on a
  second dataset (HMC) to measure real cross-dataset generalization.
- **ML vs DL under one protocol** — feature-based classifiers and raw-signal
  deep models are compared on the same data and metrics.
- **Beyond accuracy** — macro-F1, Cohen's kappa, per-class F1, confusion
  matrices, plus calibration (ECE, Brier) and SHAP / Grad-CAM explainability.

## Datasets

| Role | Dataset | Purpose |
| --- | --- | --- |
| Development | **Sleep-EDF Expanded** | Training + internal subject-wise validation |
| External test | **HMC** | PSG-to-PSG cross-dataset generalization |
| Optional extension | **DREAMT** | Wearable transfer — future work (different modality) |

> **Deviation from `docs/master_plan.md`:** the master plan's primary external
> dataset needs an access agreement that takes weeks to obtain, so this project
> uses the open-access **HMC** (still PSG-to-PSG) as the external test set
> instead. DREAMT stays optional.

Data are **not** committed to the repository. See [`data/README.md`](data/README.md)
for how to obtain and place them.

## Repository layout

```
data/         raw / processed / splits   (contents git-ignored)
src/          reusable pipeline logic (data, preprocessing, models, eval, XAI)
scripts/      thin runnable entry points (prepare data, run models, figures)
results/      generated tables, figures and logs (contents git-ignored)
tests/        tests for the leakage-sensitive logic
```

Core logic lives in `src/`; `scripts/` are thin, runnable CLIs.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Prepare data, then run a baseline:

```bash
# 0. Unpack the PhysioNet ZIP downloads placed in data/raw/ into data/raw/<dataset>/
python scripts/extract_data.py

# 1. Build model-ready epochs (see scripts for arguments)
python scripts/prepare_sleep_edf.py --raw-dir data/raw/sleep_edf --out data/processed/sleep_edf.npz

# 2. Feature correlation + importance figures (EDA; motivates selection)
python scripts/analyze_features.py --data data/processed/sleep_edf.npz

# 3. Feature-based ML with subject-wise cross-validation (+ optional selection)
python scripts/run_ml.py --data data/processed/sleep_edf.npz --model rf --feature-selection

# 4. Deep learning — Cross-Modal Transformer with temporal context
#    (auto-uses a GPU when present; add --tune for Bayesian hyperparameter search)
python scripts/run_dl.py --data data/processed/sleep_edf.npz --model transformer --context 11
python scripts/run_dl.py --data data/processed/sleep_edf.npz --model transformer --context 15 \
    --tune --search-method bayes --num-workers 2
# On Colab GPU: run notebooks/05_colab_dl_benchmark.ipynb (Sleep-EDF benchmark sweep)

# 5. External validation on a second dataset (frozen model)
python scripts/run_external.py --model-path results/logs/ml_model.pkl --external data/processed/hmc.npz
```

## Tests

The tests cover the methodology-critical logic — subject-wise splits and the
leakage guard, label mapping, filtering/resampling, training-only normalization,
leakage-safe feature selection, and metric computation:

```bash
pytest
```

## Project status

The full pipeline is implemented and tested:

- **Data:** subject-wise splitting with a leakage guard, label harmonization,
  band-pass filtering + resampling to a shared sampling rate (stored in the
  `.npz`), training-only normalization, and feature extraction (time / frequency
  / nonlinear / time-frequency — 96 features) for the ML pipeline.
- **Feature selection:** optional leakage-safe pruning (near-constant + collinear
  features, optional top-k by importance) built as pipeline steps so it re-fits
  inside every CV fold; `analyze_features.py` saves the correlation-matrix and
  feature-importance figures that motivate it.
- **Models:** feature-based ML (SVM, RF, gradient boosting, logistic — scaled
  where needed) and deep learning (1D-CNN, CNN-LSTM, and the Cross-Modal
  Transformer with modality masking for ablations and missing-modality tests).
- **Temporal context:** `--context N` turns any DL model into a hierarchical
  sequence model over N neighboring epochs (one label per epoch), with the
  single-epoch models kept as the with/without-context baseline.
- **Memory:** the DL runner streams training statistics off the memmap
  (`fit_normalizer_streaming`) and reads epochs lazily per item
  (`MemmapEpochDataset`), so it trains on the full Sleep-EDF set (~22 GB of
  signals) without loading it into RAM.
- **Imbalance:** class weights, SMOTE / random oversampling (ML), focal loss and
  balanced mini-batches (DL).
- **Tuning:** grid search (ML) and, for DL, Bayesian optimization (Optuna TPE over
  learning rate / weight decay / loss / focal gamma, `--tune --search-method bayes`)
  with random search kept as a fallback, all subject-wise.
- **Compute:** DL runs on GPU automatically (`--device auto`); `--num-workers`
  parallelizes the memmap reads. A ready-to-run Colab notebook for the Sleep-EDF
  DL benchmark sweep is in `notebooks/05_colab_dl_benchmark.ipynb`.
- **Evaluation:** macro-F1, weighted-F1, balanced accuracy, kappa, per-class
  scores, confusion matrices, one-vs-rest AUPRC / ROC-AUC.
- **Calibration:** raw-vs-calibrated ECE / Brier — isotonic/Platt (ML) and
  temperature scaling (DL) — with reliability diagrams.
- **Explainability:** SHAP (ML) and Grad-CAM (DL).
- **Validation:** internal held-out test, external cross-dataset validation, and
  LOSO robustness; figures via `make_figures.py`.

The data-preparation scripts (`prepare_sleep_edf.py`, `prepare_hmc.py`) follow
the standard dataset formats and need the actual datasets in `data/raw/` to run.
Everything else is verified on synthetic data and
by the test suite (`pytest`, 33 tests). DL hyperparameter search uses Bayesian
optimization (Optuna), as recommended in the project plan, and falls back to
random search when Optuna is not installed.
