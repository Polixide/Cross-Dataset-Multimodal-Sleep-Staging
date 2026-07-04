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

### From the PhysioNet ZIP downloads

If you downloaded the datasets as ZIP archives, just drop the `.zip` files into
`data/raw/` and let the extractor unpack each one into the right subfolder:

```bash
# put e.g. sleep-edf-database-expanded-1.0.0.zip and
# haaglanden-...-1.1.0.zip into data/raw/, then:
python scripts/extract_data.py
```

`extract_data.py` recognizes each archive from its name (or, failing that, from
the files inside it), extracts Sleep-EDF into `data/raw/sleep_edf/` and HMC into
`data/raw/hmc/`, and reports how many recording/annotation pairs it found. It
skips a dataset whose folder already contains `.edf` files (pass `--force` to
re-extract), so you can equally well extract by hand and skip this step.

## Processed format

Each `scripts/prepare_*.py` writes **two** sibling files (e.g. `sleep_edf.npz`
and `sleep_edf_x.npy`):

- `<name>_x.npy` — the epoch **signals**, shape `(n_epochs, n_channels,
  n_samples)`, stored as `float32` and **memory-mapped** on load.
- `<name>.npz` — the small metadata: `y` (integer stage labels in `[0, 4]`),
  `subjects` (subject id per epoch), `sfreq` (the sampling rate the signals were
  resampled to, in Hz), and `x_path` (the name of the `_x.npy` sibling).

`x` is kept in a standalone memmapped `.npy` because Sleep-EDF expands to ~450k
epochs (tens of GB); `load_processed_dataset` memory-maps it so no stage of the
pipeline holds the whole dataset in RAM, and `EpochStreamWriter` fills it one
recording at a time during preparation (peak memory ≈ a single recording). Point
downstream scripts at the `.npz` as before — the loader finds the `_x.npy`
sibling automatically. A legacy single `.npz` that stores `x` inline is still
read for backward compatibility.

The prepare scripts band-pass filter (default 0.3–35 Hz) and resample (default
100 Hz) each continuous recording before epoching, so all datasets share one
sampling rate; downstream feature extraction reads `sfreq` from the `.npz`.

## Rules

- Never commit raw recordings, processed arrays, or anything containing subject
  data — respect each dataset's data-use agreement.
- Keep the subject id in `subjects` consistent so subject-wise splitting works
  across preparation runs.
