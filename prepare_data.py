"""Step 1: download the MRI subset and the labels, build the cohort and the patient split.

Usage:
    python prepare_data.py --data-dir data            # downloads ~4.6 GB of NIfTI files
    python prepare_data.py --data-dir data --skip-download --image-dir /path/to/UCSF-PDGM

Writes data/cohort.csv (one row per patient: label, split, age, sex, grade, FLAIR/T1c/T2 paths).
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from idh.config import EXPECTED_PATIENTS, ExtractionConfig
from idh.data import download_metadata, download_mri, index_images, load_metadata, split_patients


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image-dir", default=None, help="folder with the NIfTI files (default: <data-dir>/mri)")
    p.add_argument("--skip-download", action="store_true", help="use images that are already on disk")
    p.add_argument("--allow-partial", action="store_true", help="keep going if some patients lack a modality")
    args = p.parse_args()

    cfg = ExtractionConfig()
    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir) if args.image_dir else data_dir / "mri"

    if not args.skip_download:
        download_mri(image_dir, cfg.modalities, token=os.environ.get("HF_TOKEN"))
    metadata_csv = download_metadata(data_dir / "UCSF-PDGM-metadata_v5.csv")

    labelled, excluded = load_metadata(metadata_csv)
    print(f"{len(labelled) + len(excluded)} metadata rows -> {len(labelled)} labelled baseline patients "
          f"({len(excluded)} excluded: {excluded['reason'].value_counts().to_dict()})")

    cohort, missing = index_images(image_dir, labelled, cfg.modalities)
    if len(missing) and not args.allow_partial:
        missing.to_csv(data_dir / "missing_modalities.csv", index=False)
        raise SystemExit(f"{len(missing)} patients lack an MRI contrast (see missing_modalities.csv). "
                         "Re-run the download or pass --allow-partial.")
    if len(cohort) != EXPECTED_PATIENTS:
        print(f"Note: {len(cohort)} patients instead of the {EXPECTED_PATIENTS} used for the reported results.")

    cohort = split_patients(cohort, cfg.test_fraction, cfg.validation_fraction, cfg.seed)
    cohort.to_csv(data_dir / "cohort.csv", index=False)
    excluded.to_csv(data_dir / "excluded.csv", index=False)

    table = pd.crosstab(cohort["split"], cohort["label"]).rename(columns={0: "wildtype", 1: "mutant"})
    print(table.loc[["train", "validation", "test"]])
    print(f"Wrote {data_dir / 'cohort.csv'}. Next: python extract_features.py")


if __name__ == "__main__":
    main()
