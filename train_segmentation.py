"""v2 step A: train the tumour segmentation model (cross-fitted) and write a predicted mask for every patient.

Usage:
    python train_segmentation.py --cohort data/cohort.csv            # GPU; ~1 h per fold on a Kaggle T4

* development patients (train + validation) -> K folds; fold k's model segments fold k
* test patients -> average of the K fold models
Writes data/pred_masks/<patient>.nii.gz, model/segmentation/fold{k}.pt (fp16),
results/v2/segmentation_dice.csv. Finished folds are reused if the run is interrupted.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from idh.config import SegConfig
from idh.segmentation import (build_cache, dice, load_weights, predict_probabilities, read_case, regions_to_brats,
                              save_mask, save_weights, train_model)
from idh.utils import get_device, seed_everything, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort", default="data/cohort.csv")
    p.add_argument("--cache-dir", default="data/seg_cache")
    p.add_argument("--masks-dir", default="data/pred_masks")
    p.add_argument("--model-dir", default="model/segmentation")
    p.add_argument("--results-dir", default="results/v2")
    p.add_argument("--epochs", type=int, default=SegConfig.epochs)
    p.add_argument("--folds", type=int, default=SegConfig.folds)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--roi", type=int, default=SegConfig.roi[0], help="training patch size (cube)")
    p.add_argument("--limit", type=int, default=None, help="use only N patients (smoke test)")
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()

    cfg = replace(SegConfig(), epochs=args.epochs, folds=args.folds, roi=(args.roi,) * 3)
    seed_everything(cfg.seed)
    device = get_device(allow_cpu=args.allow_cpu)
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": str})
    if args.limit:
        cohort = cohort.groupby("split").head(max(4, args.limit // 3)).reset_index(drop=True)
    model_dir, masks_dir, results = Path(args.model_dir), Path(args.masks_dir), Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    print(f"Caching {len(cohort)} volumes -> {args.cache_dir}")
    build_cache(cohort, args.cache_dir)

    development = cohort[cohort["split"] != "test"].reset_index(drop=True)
    test = cohort[cohort["split"] == "test"].reset_index(drop=True)
    splitter = StratifiedKFold(cfg.folds, shuffle=True, random_state=cfg.seed)
    fold_of = np.zeros(len(development), dtype=int)
    for k, (_, held) in enumerate(splitter.split(development, development["label"]), 1):
        fold_of[held] = k
    development.assign(seg_fold=fold_of)[["patient_id", "seg_fold"]].to_csv(results / "segmentation_folds.csv", index=False)

    rows, models = [], []
    for k in range(1, cfg.folds + 1):
        weights = model_dir / f"fold{k}.pt"
        train_ids = development.loc[fold_of != k, "patient_id"]
        if weights.exists():
            print(f"Fold {k}: reusing {weights}")
            model = load_weights(weights, cfg, device)
        else:
            print(f"Fold {k}: training on {len(train_ids)} patients for {cfg.epochs} epochs")
            model = train_model(train_ids, args.cache_dir, cfg, device, cfg.seed + k, args.workers)
            save_weights(model, weights)
            model = load_weights(weights, cfg, device)  # use exactly the saved (fp16-rounded) weights
        models.append(model)
        for pid in development.loc[fold_of == k, "patient_id"]:
            case = read_case(args.cache_dir, pid)
            label = regions_to_brats(predict_probabilities([model], case["image"], cfg, device) > 0.5)
            save_mask(label, case, masks_dir / f"{pid}.nii.gz")
            rows.append({"patient_id": pid, "split": "development", "fold": k, **dice(label, case["label"])})
        print(f"  fold {k} held-out Dice: " + ", ".join(
            f"{r} {np.mean([x[r] for x in rows if x['fold'] == k]):.3f}" for r in ("WT", "TC", "ET")))

    print(f"Segmenting {len(test)} test patients with the {cfg.folds}-model ensemble")
    for pid in test["patient_id"]:
        case = read_case(args.cache_dir, pid)
        label = regions_to_brats(predict_probabilities(models, case["image"], cfg, device) > 0.5)
        save_mask(label, case, masks_dir / f"{pid}.nii.gz")
        rows.append({"patient_id": pid, "split": "test", "fold": 0, **dice(label, case["label"])})

    scores = pd.DataFrame(rows)
    scores.to_csv(results / "segmentation_dice.csv", index=False)
    summary = scores.groupby("split")[["WT", "TC", "ET"]].agg(["mean", "median"]).round(3)
    print(summary)
    write_json(results / "segmentation_summary.json",
               {"config": asdict(cfg), "dice": {s: g[["WT", "TC", "ET"]].mean().to_dict()
                                                for s, g in scores.groupby("split")}})
    print(f"Masks in {masks_dir}/. Next: python extract_tumour_features.py")


if __name__ == "__main__":
    main()
