"""v2 step B: tumour-focused DINOv2 features and hand-crafted tumour descriptors from a mask.

Usage:
    python extract_tumour_features.py --masks pred      # masks from train_segmentation.py (the real pipeline)
    python extract_tumour_features.py --masks gt        # expert masks (upper bound; not available for new patients)

For every patient: the slices that contain tumour are cropped to a square box around the tumour,
encoded by frozen DINOv2-small (as in v1, but now the tumour fills the image), and ~40 descriptors
(volumes, enhancement, T2-FLAIR mismatch, location, shape) are computed from the mask.
Writes <out-dir>/patient_manifest.csv, features/<patient>.npz and descriptors.csv. Resumable.
GPU recommended (~10 min); a CPU works too (~1 h).
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from idh.config import ExtractionConfig, TumourConfig
from idh.features import encode_volume, load_encoder, model_digest
from idh.preprocessing import prepare_volume
from idh.tumour import load_mask, tumour_descriptors, tumour_slices
from idh.utils import get_device, seed_everything, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="data/cohort.csv")
    p.add_argument("--masks", choices=["pred", "gt"], default="pred")
    p.add_argument("--masks-dir", default="data/pred_masks")
    p.add_argument("--out-dir", default=None, help="default: features_tumour_<masks>")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    cfg, tcfg = ExtractionConfig(), TumourConfig()
    seed_everything(cfg.seed)
    device = get_device(allow_cpu=args.allow_cpu)
    out = Path(args.out_dir or f"features_tumour_{args.masks}")
    (out / "features").mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    encoder = load_encoder(cfg, device, token=os.environ.get("HF_TOKEN"))
    write_json(out / "extraction.json", {"config": asdict(cfg), "tumour_config": asdict(tcfg), "masks": args.masks,
                                         "encoder_sha256": model_digest(encoder)})

    rows, descriptors = [], []
    for row in tqdm(cohort.to_dict("records"), desc=f"Tumour features ({args.masks} masks)"):
        pid = row["patient_id"]
        npz, info = out / "features" / f"{pid}.npz", out / "features" / f"{pid}.json"
        if not (npz.exists() and info.exists()):
            mask_path = row["seg"] if args.masks == "gt" else Path(args.masks_dir) / f"{pid}.nii.gz"
            prepared = prepare_volume(row, cfg.modalities, cfg.crop_margin_voxels)
            mask = load_mask(mask_path, prepared["arrays"][0].shape)
            focused, found = tumour_slices(prepared, mask, tcfg)
            cls, patch = encode_volume(focused, encoder, cfg, device)
            tmp = npz.with_name(npz.name + ".part")
            with tmp.open("wb") as f:
                np.savez_compressed(f, cls_features=cls, patch_mean_features=patch, slice_indices=focused["indices"],
                                    slice_z_mm=focused["z_mm"], slice_fraction=focused["fractions"],
                                    nonzero_area_fraction=focused["area"])
            tmp.replace(npz)
            info.write_text(json.dumps({"tumour_found": found, "n_slices": int(len(cls)),
                                        "descriptors": tumour_descriptors(row, mask)}))
        meta = json.loads(info.read_text())
        rows.append({"patient_id": pid, "n_slices": meta["n_slices"], "tumour_found": meta["tumour_found"],
                     "feature_file": f"features/{pid}.npz"})
        descriptors.append({"patient_id": pid, **meta["descriptors"]})

    keep = ["patient_id", "label", "split", "age", "sex", "grade"]
    manifest = cohort[keep].merge(pd.DataFrame(rows), on="patient_id", validate="one_to_one")
    manifest.to_csv(out / "patient_manifest.csv", index=False)
    pd.DataFrame(descriptors).to_csv(out / "descriptors.csv", index=False)
    print(f"{len(manifest)} patients, {int(manifest['n_slices'].sum())} tumour slices "
          f"(median {int(manifest['n_slices'].median())}/patient), no tumour found for "
          f"{int((~manifest['tumour_found']).sum())}. Next: python run_cv.py --features {out}")


if __name__ == "__main__":
    main()
