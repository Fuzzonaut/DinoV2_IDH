"""Turn the archive written by the original Kaggle notebook into this repo's layout.

The notebook's last cell saves /kaggle/working/idh_dinov2_auto_slices_v1.zip (~220 MB) with the cached
features and the finished experiment. This script copies

    <experiment>/final_model/head.pt + prediction_manifest.json  ->  model/head.pt + model/config.json
    <features>/ (patient_manifest.csv, features/*.npz, ...)      ->  features/        (with --features)

so that predict.py works with the trained head, and search_slices.py / train.py / evaluate.py can be
re-run without the MRI volumes or a GPU.

Usage:
    python import_kaggle_run.py idh_dinov2_auto_slices_v1.zip --features
"""

import argparse
import shutil
import zipfile
from pathlib import Path

from idh.training import load_head
from idh.utils import read_json, sha256_file, write_json

EXPERIMENT = "idh_dinov2_auto_slices_v1"
FEATURES = "idh_dinov2_frozen_v1"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archive", help="the Kaggle .zip, or a folder it was extracted to")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--features", action="store_true", help="also copy the feature cache to --features-dir")
    p.add_argument("--features-dir", default="features")
    args = p.parse_args()

    source = Path(args.archive)
    if source.suffix == ".zip":
        target = source.with_suffix("")
        print(f"Extracting {source} -> {target}")
        with zipfile.ZipFile(source) as archive:
            archive.extractall(target)
        source = target

    experiment = source / EXPERIMENT
    manifest = read_json(experiment / "prediction_manifest.json")
    head_path = experiment / manifest["head_file"]
    if sha256_file(head_path) != manifest["head_sha256"]:
        raise SystemExit("head.pt does not match the checksum in prediction_manifest.json.")
    load_head(head_path)  # fails early if the checkpoint is not compatible

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(head_path, model_dir / "head.pt")
    write_json(model_dir / "config.json", {
        "extraction_config": manifest["extraction_config"],
        "encoder_sha256": manifest["feature_identity"]["model_state_sha256"],
        "feature_kind": manifest["feature_kind"],
        "selected_count": manifest["selected_count"],
        "slice_range": manifest["slice_range"],
        "threshold": manifest["threshold"],
        "threshold_objective": "balanced accuracy on the validation split",
        "class_names": manifest["classnames"],
        "training_epochs": manifest["training_epochs"],
        "train_mutant_fraction": manifest["train_prevalence"],
    })
    print(f"Model: {model_dir}/head.pt ({manifest['selected_candidate']} slices, threshold {manifest['threshold']:.3f})")

    if args.features:
        destination = Path(args.features_dir)
        shutil.copytree(source / FEATURES, destination, dirs_exist_ok=True)
        print(f"Features: {destination}/ ({len(list((destination / 'features').glob('*.npz')))} patients)")


if __name__ == "__main__":
    main()
