"""v2 step A (fast): tumour masks from a pretrained BraTS segmentation model - no training.

Usage:
    python segment_pretrained.py --cohort data/cohort.csv        # ~4 s/patient on a laptop CPU, <1 s on a GPU

Uses MONAI's `brats_mri_segmentation` bundle (SegResNet trained on BraTS 2018, which does not include
UCSF-PDGM), so every patient - train, validation and test - is segmented by a model that never saw
them. Dice against the UCSF-PDGM expert masks is reported for every patient.
Writes data/pred_masks/<patient>.nii.gz and results/v2/segmentation_dice.csv. Resumable.
(train_segmentation.py trains a cross-fitted model from scratch instead; slower, needs a GPU.)
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from idh.segmentation import dice, load_case, load_pretrained, predict_pretrained, regions_to_brats, save_mask
from idh.utils import get_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="data/cohort.csv")
    p.add_argument("--masks-dir", default="data/pred_masks")
    p.add_argument("--results-dir", default="results/v2")
    p.add_argument("--flip-tta", action="store_true", help="also predict on the flipped volume and average (2x time)")
    args = p.parse_args()

    device = get_device()
    model = load_pretrained(device)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    masks, results = Path(args.masks_dir), Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    started, rows = time.time(), []
    for row in tqdm(cohort.to_dict("records"), desc=f"Segmenting on {device}"):
        has_truth = isinstance(row.get("seg"), str)
        case = load_case(row, with_label=has_truth)
        out = masks / f"{row['patient_id']}.nii.gz"
        if out.exists():  # resume: reuse the saved mask
            full = np.asarray(nib.as_closest_canonical(nib.load(str(out))).dataobj)
            (x0, x1), (y0, y1), (z0, z1) = case["bbox"]
            label = full[x0:x1, y0:y1, z0:z1]
        else:
            label = regions_to_brats(predict_pretrained(model, case["image"], device, args.flip_tta) > 0.5)
            save_mask(label, case, out)
        if has_truth:
            rows.append({"patient_id": row["patient_id"], "split": row["split"], **dice(label, case["label"])})

    if rows:
        scores = pd.DataFrame(rows)
        scores.to_csv(results / "segmentation_dice.csv", index=False)
        summary = scores[["WT", "TC", "ET"]].agg(["mean", "median"]).round(3)
        print(summary)
        write_json(results / "segmentation_summary.json", {
            "model": "MONAI brats_mri_segmentation (pretrained, BraTS 2018)", "flip_tta": args.flip_tta,
            "dice_mean": scores[["WT", "TC", "ET"]].mean().to_dict(),
            "dice_median": scores[["WT", "TC", "ET"]].median().to_dict(),
            "minutes": (time.time() - started) / 60})
    print(f"Masks in {masks}/. Next: python extract_tumour_features.py --masks pred")


if __name__ == "__main__":
    main()
