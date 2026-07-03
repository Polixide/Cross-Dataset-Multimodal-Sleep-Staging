# Data

Datasets are **not** stored in this repository — they are large and/or
access-controlled. This folder only holds the structure; its contents are
git-ignored (except these placeholders).

## Layout

```
data/
├── raw/         # original downloads, one subfolder per dataset (untracked)
│   ├── sleep_edf/
│   ├── hmc/
│   └── dreamt/
├── processed/   # model-ready .npz files produced by scripts/prepare_*.py
│   ├── sleep_edf.npz
│   └── hmc.npz
└── splits/      # saved subject-wise split indices for reproducibility
```

The `prepare_*.py` scripts search recursively, so you can leave a download in its
original subfolder structure (e.g. `data/raw/hmc/recordings/...`).

## Obtaining the datasets

- **Sleep-EDF Expanded** — PhysioNet (Sleep-EDF Database Expanded); open access.
  Needs the paired `*-PSG.edf` + `*-Hypnogram.edf` files.
- **HMC** — PhysioNet (Haaglanden Medisch Centrum sleep staging database); open
  access, no agreement. Needs the `SNxxx.edf` + `SNxxx_sleepscoring.edf` pairs.
  Used as the external test set.
- **DREAMT** — PhysioNet (optional wearable extension; different modality, treated
  as future work).

Download each dataset from its official source, then place the raw files under
`data/raw/<dataset>/`.

## Processed format

Each `scripts/prepare_*.py` writes a single compressed `.npz` with three aligned
arrays — `x` (epoch signals, shape `(n_epochs, n_channels, n_samples)`; or
`(n_epochs, n_features)` for the feature-based pipeline), `y` (integer stage
labels in `[0, 4]`), and `subjects` (subject id per epoch) — plus a scalar
`sfreq` (the sampling rate the signals were resampled to, in Hz). The prepare
scripts band-pass filter (default 0.3–35 Hz) and resample (default 100 Hz) each
continuous recording before epoching, so all datasets share one sampling rate;
downstream feature extraction reads `sfreq` from the `.npz`.

## Rules

- Never commit raw recordings, processed arrays, or anything containing subject
  data — respect each dataset's data-use agreement.
- Keep the subject id in `subjects` consistent so subject-wise splitting works
  across preparation runs.
