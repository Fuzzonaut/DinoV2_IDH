"""Step 3: choose the number of slices per patient by cross-validation on the training patients.

Usage:
    python search_slices.py --features features
    python search_slices.py --features features --cv-folds 5 --cv-repeats 3   # less noisy selection

Runs in about a minute on a CPU (only the small attention head is trained).
Writes results/slice_search.csv, slice_search_folds.csv, cv_predictions.csv,
slice_selection.json and slice_search_curve.png.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from idh.config import SearchConfig
from idh.features import load_features, load_manifest
from idh.plots import plot_slice_search
from idh.selection import search_slice_counts
from idh.utils import get_device, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="features")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--cv-folds", type=int, default=SearchConfig.cv_folds)
    p.add_argument("--cv-repeats", type=int, default=SearchConfig.cv_repeats)
    p.add_argument("--feature-kind", default=SearchConfig.feature_kind, choices=["cls", "patch_mean", "concat"])
    args = p.parse_args()

    settings = replace(SearchConfig(), cv_folds=args.cv_folds, cv_repeats=args.cv_repeats,
                       feature_kind=args.feature_kind)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.features)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    store = load_features(args.features, train, settings.feature_kind)
    print(f"{len(train)} training patients, {settings.cv_repeats} x {settings.cv_folds}-fold CV")

    selection, results, folds, oof = search_slice_counts(train, store, settings, get_device())
    results.sort_values("mean_cv_auroc", ascending=False).to_csv(out / "slice_search.csv", index=False)
    folds.to_csv(out / "slice_search_folds.csv", index=False)
    oof.to_csv(out / "cv_predictions.csv", index=False)
    write_json(out / "slice_selection.json", {**selection, "settings": asdict(settings)})
    plot_slice_search(results, selection["selected_candidate"], out / "slice_search_curve.png")

    print(f"Selected: {selection['selected_candidate']} (CV AUROC {selection['selected_cv_auroc']:.4f}), "
          f"final model trains for {selection['final_training_epochs']} epochs. Next: python train.py")


if __name__ == "__main__":
    main()
