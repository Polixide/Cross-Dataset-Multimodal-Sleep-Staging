"""Extract downloaded dataset ZIP archives into data/raw/<dataset>/.

Drop the PhysioNet ZIP downloads anywhere under ``data/raw/`` (or pass their
paths explicitly) and this script unpacks each one into the folder the
``prepare_*.py`` scripts expect:

    data/raw/sleep_edf/   <- Sleep-EDF Expanded zip
    data/raw/hmc/         <- HMC (Haaglanden Medisch Centrum) zip
    data/raw/isruc/       <- ISRUC-Sleep Cohort I zip

Each archive is recognized automatically from its file name and, if that is
inconclusive, from the names of the files it contains (a Sleep-EDF archive holds
``*-PSG.edf`` files; an HMC archive holds ``*_sleepscoring.edf`` files; an ISRUC
archive holds ``*.rec`` files). The
prepare scripts search recursively, so the archive's own subfolders
(``sleep-cassette/``, ``recordings/``, ...) are kept as-is; no flattening needed.

If a target folder already contains ``.edf`` files the archive is skipped, so
you can also just extract by hand and run the prepare scripts directly. Use
``--force`` to re-extract.

Usage:
    # auto-detect any *.zip sitting in data/raw/
    python scripts/extract_data.py

    # or point at specific archives
    python scripts/extract_data.py \
        --sleep-edf-zip data/raw/sleep-edf-database-expanded-1.0.0.zip \
        --hmc-zip data/raw/haaglanden-medisch-centrum-sleep-staging-database-1.1.0.zip

ISRUC-Sleep is distributed as one .rar per subject (not a single zip). If you
have those, extract them by hand into data/raw/isruc/<subject>/ and run
prepare_isruc.py directly; this helper only unpacks a single ISRUC .zip if you
happen to have one.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, get_logger

logger = get_logger("extract_data")

# Target subfolder (under --raw-dir) for each recognized dataset.
DATASET_DIRS = {"sleep_edf": "sleep_edf", "hmc": "hmc", "isruc": "isruc"}


def classify_zip(zip_path):
    """Guess which dataset an archive holds: "sleep_edf", "hmc", "isruc", or None.

    Tries the file name first (cheap), then falls back to inspecting the member
    names, which only reads the zip's central directory (no full extraction).
    """
    name = zip_path.name.lower()
    if "sleep-edf" in name or "sleep_edf" in name or "sleepedf" in name:
        return "sleep_edf"
    if "haaglanden" in name or "hmc" in name:
        return "hmc"
    if "isruc" in name:
        return "isruc"

    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [member.lower() for member in archive.namelist()]
    except zipfile.BadZipFile:
        logger.warning("%s is not a valid zip file.", zip_path.name)
        return None

    if any(member.endswith("-psg.edf") for member in members):
        return "sleep_edf"
    if any(member.endswith("_sleepscoring.edf") for member in members):
        return "hmc"
    if any(member.endswith(".rec") for member in members):
        return "isruc"
    return None


def extract_zip(zip_path, dest_dir, force=False):
    """Extract one archive into dest_dir; skip if it already holds .edf files."""
    dest_dir = ensure_dir(dest_dir)
    if any(dest_dir.rglob("*.edf")) and not force:
        logger.info("Skipping %s: %s already contains .edf files (use --force to re-extract).",
                    zip_path.name, dest_dir)
        return

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        logger.info("Extracting %s (%d entries) -> %s ...", zip_path.name, len(members), dest_dir)
        archive.extractall(dest_dir)
    logger.info("Finished extracting %s.", zip_path.name)


def verify(dataset, dest_dir):
    """Log how many usable file pairs landed in dest_dir; return True if usable."""
    dest_dir = Path(dest_dir)
    if dataset == "sleep_edf":
        psg = list(dest_dir.rglob("*-PSG.edf"))
        hypnograms = list(dest_dir.rglob("*-Hypnogram.edf"))
        logger.info("sleep_edf: %d PSG file(s), %d hypnogram file(s).", len(psg), len(hypnograms))
        return bool(psg) and bool(hypnograms)
    if dataset == "hmc":
        signals = [f for f in dest_dir.rglob("*.edf") if not f.name.endswith("_sleepscoring.edf")]
        scoring = list(dest_dir.rglob("*_sleepscoring.edf"))
        logger.info("hmc: %d signal file(s), %d scoring file(s).", len(signals), len(scoring))
        return bool(signals) and bool(scoring)
    if dataset == "isruc":
        recordings = list(dest_dir.rglob("*.rec"))
        hypnograms = list(dest_dir.rglob("*_1.txt")) + list(dest_dir.rglob("*_2.txt"))
        logger.info("isruc: %d recording(s), %d expert hypnogram(s).", len(recordings), len(hypnograms))
        return bool(recordings) and bool(hypnograms)
    return False


def main():
    parser = argparse.ArgumentParser(description="Extract dataset ZIP archives into data/raw/.")
    parser.add_argument("--zip-dir", default="data/raw",
                        help="Folder scanned for *.zip when explicit paths are not given.")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Base folder; each dataset is extracted to <raw-dir>/<dataset>/.")
    parser.add_argument("--sleep-edf-zip", default=None, help="Explicit path to the Sleep-EDF zip.")
    parser.add_argument("--hmc-zip", default=None, help="Explicit path to the HMC zip.")
    parser.add_argument("--isruc-zip", default=None, help="Explicit path to an ISRUC-Sleep zip (if bundled).")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if the target already contains .edf files.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    jobs = {}  # dataset -> zip path (explicit args take precedence)
    if args.sleep_edf_zip:
        jobs["sleep_edf"] = Path(args.sleep_edf_zip)
    if args.hmc_zip:
        jobs["hmc"] = Path(args.hmc_zip)
    if args.isruc_zip:
        jobs["isruc"] = Path(args.isruc_zip)

    # Fill in any dataset not given explicitly by scanning --zip-dir.
    for zip_path in sorted(Path(args.zip_dir).glob("*.zip")):
        dataset = classify_zip(zip_path)
        if dataset is None:
            logger.warning("Could not identify %s; skipping. Pass it with "
                           "--sleep-edf-zip / --hmc-zip if it is a dataset archive.", zip_path.name)
            continue
        jobs.setdefault(dataset, zip_path)

    if not jobs:
        raise FileNotFoundError(
            f"No dataset zips found. Put the downloads in {args.zip_dir}/ "
            "or pass --sleep-edf-zip / --hmc-zip / --isruc-zip."
        )

    all_ok = True
    for dataset, zip_path in jobs.items():
        if not zip_path.exists():
            logger.error("Zip not found: %s", zip_path)
            all_ok = False
            continue
        dest_dir = raw_dir / DATASET_DIRS[dataset]
        extract_zip(zip_path, dest_dir, force=args.force)
        if not verify(dataset, dest_dir):
            logger.warning("%s: expected .edf files not found under %s after extraction.",
                           dataset, dest_dir)
            all_ok = False

    logger.info("Extraction complete. Next, build the model-ready epochs:")
    if "sleep_edf" in jobs:
        logger.info("  python scripts/prepare_sleep_edf.py --raw-dir %s --out data/processed/sleep_edf.npz",
                    (raw_dir / "sleep_edf").as_posix())
    if "hmc" in jobs:
        logger.info("  python scripts/prepare_hmc.py --raw-dir %s --out data/processed/hmc.npz",
                    (raw_dir / "hmc").as_posix())
    if "isruc" in jobs:
        logger.info("  python scripts/prepare_isruc.py --raw-dir %s --out data/processed/isruc.npz",
                    (raw_dir / "isruc").as_posix())
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
