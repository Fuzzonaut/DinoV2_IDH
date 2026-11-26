"""Predict the IDH status of one patient from co-registered, skull-stripped FLAIR / T1c / T2 volumes.

Usage:
    python predict.py --flair flair.nii.gz --t1c t1c.nii.gz --t2 t2.nii.gz
    python predict.py --patient-dir data/mri/UCSF-PDGM-0021      # finds *_FLAIR / *_T1c / *_T2 files

Downloads DINOv2-small (~90 MB) on first use. Research code, not a medical device.
"""

import argparse
from pathlib import Path

from idh.data import MODALITY_SUFFIXES
from idh.inference import IDHPredictor


def find_volumes(folder):
    paths = {}
    for modality, suffixes in MODALITY_SUFFIXES.items():
        for suffix in suffixes:
            hits = sorted(Path(folder).glob(f"*_{suffix}.nii*"))
            if hits:
                paths[modality] = str(hits[0])
                break
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flair")
    p.add_argument("--t1c")
    p.add_argument("--t2")
    p.add_argument("--patient-dir")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.patient_dir:
        paths = find_volumes(args.patient_dir)
    else:
        paths = {"FLAIR": args.flair, "T1c": args.t1c, "T2": args.t2}
        paths = {k: v for k, v in paths.items() if v}
    if len(paths) != 3:
        p.error(f"Need FLAIR, T1c and T2; found {sorted(paths)}.")

    result = IDHPredictor(args.model_dir, device=args.device).predict(paths)
    print(f"Prediction: {result['predicted_class']}")
    print(f"Score:      {result['score']:.3f} (threshold {result['threshold']:.3f}; not a calibrated probability)")
    print(f"Slices:     {result['n_slices_used']}")
    top = sorted(zip(result["attention"], result["slice_index"]), reverse=True)[:5]
    print("Most attended axial slices:", ", ".join(f"z={z} ({w:.1%})" for w, z in top))


if __name__ == "__main__":
    main()
