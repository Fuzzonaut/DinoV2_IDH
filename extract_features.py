"""Step 2: encode every nonempty axial slice of every patient with frozen DINOv2 and cache the features.

Usage:
    python extract_features.py --cohort data/cohort.csv --out-dir features

About 14 minutes on a Kaggle T4/P100 GPU for 495 patients (~70k slices, ~200 MB of features).
Interrupted runs resume: finished patients are skipped.
"""

import argparse
import os

import pandas as pd

from idh.config import ExtractionConfig
from idh.features import extract_cohort, load_encoder
from idh.utils import get_device, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="data/cohort.csv")
    p.add_argument("--out-dir", default="features")
    p.add_argument("--allow-cpu", action="store_true", help="run on the CPU (slow: hours instead of minutes)")
    args = p.parse_args()

    cfg = ExtractionConfig()
    seed_everything(cfg.seed)
    device = get_device(allow_cpu=args.allow_cpu)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})

    encoder = load_encoder(cfg, device, token=os.environ.get("HF_TOKEN"))
    print(f"Encoder: {cfg.model_id} (frozen, {encoder.feature_dim}-d features) on {device}")
    manifest = extract_cohort(cohort, encoder, cfg, device, args.out_dir)
    print(f"{len(manifest)} patients, {manifest['n_slices'].sum()} slices -> {args.out_dir}. "
          "Next: python search_slices.py")


if __name__ == "__main__":
    main()
