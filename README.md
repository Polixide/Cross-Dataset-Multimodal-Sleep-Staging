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
- **External validation** — a model trained on Sleep-EDF is tested, frozen, on
  HMC to measure real cross-dataset generalization.
- **ML vs DL under one protocol** — feature-based classifiers and raw-signal
  deep models are compared on the same data and metrics.
- **Beyond accuracy** — macro-F1, Cohen's kappa, per-class F1, confusion
  matrices, plus calibration (ECE, Brier) and SHAP / Grad-CAM explainability.

## Datasets

| Role | Dataset | Purpose |
| --- | --- | --- |
| Development | **Sleep-EDF Expanded** | Training + internal subject-wise validation |
| External test | **HMC** / **SHHS** | PSG-to-PSG cross-dataset generalization |
| Optional extension | **DREAMT** | Wearable transfer — future work (different modality) |

Data are **not** committed to the repository. See [`data/README.md`](data/README.md)
for how to obtain and place them.

## Repository layout

```
data/         raw / processed / splits   (contents git-ignored)
src/          reusable pipeline logic (data, preprocessing, models, eval, XAI)
scripts/      runnable entry points split into data/, ml/, dl/, shared/
results/      generated tables, figures and logs (contents git-ignored)
tests/        tests for the leakage-sensitive logic
```

Core logic is split into `src/common/`, `src/ml/`, and `src/dl/`; runnable CLIs
mirror that separation under `scripts/`.

La descrizione metodologica completa della parte ML è disponibile in
[`ML_WORKFLOW_CHECK.md`](ML_WORKFLOW_CHECK.md).

## Getting started

```bash
python -m venv venv && source venv/bin/activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Prepare data, then run a baseline:

```bash
# 1. Build model-ready epochs (see scripts for arguments)
python scripts/data/prepare_sleep_edf.py --raw-dir data/raw/sleep_edf --out data/processed/sleep_edf.npz

# 2. Feature-based ML with subject-wise cross-validation
python scripts/ml/run_ml.py --data data/processed/sleep_edf.npz --model rf

# If --calibration is omitted, run_ml.py saves both *_raw and *_sigmoid outputs
# after one shared tuning/CV pass. Use --calibration none/sigmoid/isotonic to
# save only one variant.

# 3. Deep learning — Cross-Modal Transformer with temporal context
python scripts/dl/run_dl.py --data data/processed/sleep_edf.npz --model transformer --context 11

# 4. External validation on a second dataset (frozen model)
python scripts/ml/run_external.py --model-path results/logs/ml_model.pkl --external data/processed/hmc.npz
```

Run the complete ML workflow after preparing both Sleep-EDF and HMC feature
files. It performs constrained train-only GroupKFold tuning, class weighting,
half-SMOTE on the best model, HMC validation, reporting, and final LOSO:

```bash
python scripts/ml/run_ml_workflow.py
```

## Classical ML pipeline

### Required inputs

The training workflow reads precomputed feature matrices and never modifies
them:

```text
data/processed/sleep_edf_features.npz  # development dataset
data/processed/hmc_features.npz        # frozen external dataset
```

Each NPZ contains `x`, `y`, `subjects`, `feature_names`, and `sfreq`. The current
matrices contain 96 handcrafted features per 30-second epoch and use five shared
classes: Wake, N1, N2, N3, and REM.

### Leakage-safe evaluation protocol

Sleep-EDF is split once by subject:

```text
100 subjects
  -> 70 training subjects
  -> 15 calibration/validation subjects
  -> 15 held-out test subjects
```

All epochs belonging to one subject remain in the same partition. Hyperparameter
tuning uses 5-fold `GroupKFold` only within the 70 training subjects. In each
fold, approximately 56 subjects are used for fitting and 14 unseen subjects for
CV scoring. The fixed validation set is reserved for probability calibration;
the test set is touched only for final evaluation.

### Models and imbalance handling

The benchmark compares multinomial Logistic Regression, Random Forest, and
XGBoost. Logistic Regression receives standardized features; tree models use
the original feature scale.

Balanced class weights are the primary imbalance strategy. For multiclass
XGBoost, the same fold-local class weights are applied to individual training
rows because XGBoost has no native multiclass `class_weight` argument.

The optional `smote_half` ablation applies SMOTE inside each training fold only.
Each minority class is capped at 50% of the majority-class count, avoiding the
much larger dataset produced by full SMOTE. Synthetic rows exist only in memory:
validation, test, HMC, subject identifiers, and source NPZ files are unchanged.

### Constrained tuning and overfitting

For every hyperparameter configuration, the code computes:

```text
overfitting gap = mean train Macro-F1 - mean validation Macro-F1
```

Configurations with a gap above `0.10` are rejected. Among eligible
configurations, the highest validation Macro-F1 is selected. If none satisfies
the constraint, the smallest-gap configuration is used as an explicitly logged
fallback. This threshold is a project heuristic, so internal-test and external
HMC performance must still be reported.

### Final fit, calibration, and testing

The selected estimator is fitted on all 70 training subjects. Its parameters
are frozen and sigmoid/Platt calibration is fitted on the 15 validation
subjects. Both raw and calibrated predictions are evaluated on the 15 held-out
test subjects.

Metrics include accuracy, balanced accuracy, Macro-F1, weighted-F1, per-class
F1, Cohen's kappa, one-vs-rest ROC-AUC/AUPRC, ECE, and multiclass Brier score.
The frozen model is then evaluated on HMC without retuning or recalibration.

### Commands

Complete workflow: three class-weight models, constrained selection,
half-SMOTE on the winner, HMC, figures, SHAP, and final LOSO:

```powershell
python scripts/ml/run_ml_workflow.py
```

XGBoost half-SMOTE using already tuned parameters, while retaining 5-fold CV
to measure the overfitting gap:

```powershell
python scripts/ml/run_ml.py `
  --data data/processed/sleep_edf_features.npz `
  --model xgb `
  --balance smote_half `
  --params-from results/logs/ml_metrics_xgb.json `
  --folds 5 `
  --calibration sigmoid `
  --json-out results/logs/ml_metrics_xgb_smote_half.json `
  --model-out results/logs/ml_model_xgb_smote_half.pkl `
  --probs-out results/logs/ml_test_probs_xgb_smote_half.npz
```

Add `--skip-cv` for one final fit only. This is faster, but no overfitting gap
can then be estimated for that run.

LOSO robustness with tuned XGBoost and class weighting:

```powershell
python scripts/ml/run_ml.py `
  --data data/processed/sleep_edf_features.npz `
  --model xgb `
  --balance class_weight `
  --loso `
  --params-from results/logs/ml_metrics_xgb.json `
  --json-out results/logs/loso_metrics_xgb_class_weight.json
```

LOSO fits one model per held-out subject and displays progress via `tqdm`. It is
a robustness analysis; comparison with GroupKFold, internal test, and HMC is
more informative than treating LOSO as a separate overfitting gap.

### Outputs

```text
results/tables/ml_train_sleep.csv           train/internal comparison table
results/tables/ml_workflow_summary.csv      complete workflow summary
results/logs/ml_metrics_<run>.json          detailed metrics and protocol
results/logs/ml_model_<run>.pkl             calibrated fitted estimator
results/logs/ml_test_probs_<run>.npz        raw/calibrated test probabilities
results/figures/<run>/                      CM, F1, ROC, PR, reliability, SHAP
```

SHAP currently stores the global feature-importance PNG. Numerical SHAP arrays
are not yet persisted and must be recomputed for alternative SHAP plots.

## Tests

The tests cover the methodology-critical logic — subject-wise splits and the
leakage guard, label mapping, training-only normalization, and metric
computation:

```bash
pytest
```

## Project status

The full pipeline is implemented and tested:

- **Data:** subject-wise splitting with a leakage guard, label harmonization,
  training-only normalization, and feature extraction (time / frequency /
  nonlinear / time-frequency) for the ML pipeline.
- **Models:** feature-based ML (multinomial logistic regression, RF, XGBoost — scaled
  where needed) and deep learning (1D-CNN, CNN-LSTM, and the Cross-Modal
  Transformer with modality masking for ablations and missing-modality tests).
- **Temporal context:** `--context N` turns any DL model into a hierarchical
  sequence model over N neighboring epochs (one label per epoch), with the
  single-epoch models kept as the with/without-context baseline.
- **Imbalance:** fold-local balanced class weights, half-SMOTE / random oversampling
  ablations (ML), focal loss and
  balanced mini-batches (DL).
- **Tuning:** grid search (ML) and random search (DL), both subject-wise.
- **Evaluation:** macro-F1, weighted-F1, balanced accuracy, kappa, per-class
  scores, confusion matrices, one-vs-rest AUPRC / ROC-AUC.
- **Calibration:** raw-vs-calibrated ECE / Brier — isotonic/Platt (ML) and
  temperature scaling (DL) — with reliability diagrams.
- **Explainability:** SHAP (ML) and Grad-CAM (DL).
- **Validation:** internal held-out test, external cross-dataset validation, and
  LOSO robustness; figures via `make_figures.py`.

The data-preparation scripts under `scripts/data/` follow the standard dataset
formats and need the actual
datasets in `data/raw/` to run. Everything else is verified on synthetic data and
by the test suite (`pytest`, 37 tests). One documented design choice: DL
hyperparameter search uses random search rather than Bayesian optimization, to
avoid an extra dependency.
