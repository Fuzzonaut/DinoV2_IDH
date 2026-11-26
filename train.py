"""Step 4: train the final head on all training patients and pick the decision threshold on validation.

Usage:
    python train.py --features features --selection results/slice_selection.json

Uses the slice count and epoch count chosen in step 3. Validation patients are not used for
fitting; they only choose the threshold (maximum balanced accuracy).
Writes model/head.pt + model/config.json and results/metrics_validation.json.
"""

import argparse
from pathlib import Path

import pandas as pd

from idh.config import CLASS_NAMES, ExtractionConfig, SearchConfig
from idh.features import load_features, load_manifest, make_bags
from idh.metrics import choose_threshold, summarize
from idh.training import fit_head, predict_proba, save_head
from idh.utils import get_device, read_json, write_json


def extraction_info(feature_dir):
    """Extraction settings + encoder checksum, from this repo's cache or from an original Kaggle feature pack."""
    feature_dir = Path(feature_dir)
    if (feature_dir / "extraction.json").exists():
        info = read_json(feature_dir / "extraction.json")
        return info["config"], info.get("encoder_sha256")
    config = read_json(feature_dir / "extraction_config.json")
    identity = read_json(feature_dir / "run_identity.json")["feature_identity"]
    return config, identity.get("model_state_sha256")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="features")
    p.add_argument("--selection", default="results/slice_selection.json")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    selection = read_json(args.selection)
    settings = SearchConfig(**{**selection["settings"], "counts": tuple(selection["settings"]["counts"])})
    device = get_device()

    manifest = load_manifest(args.features)
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    validation = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    development = pd.concat([train, validation], ignore_index=True)
    store = load_features(args.features, development, settings.feature_kind)
    bags = make_bags(development["patient_id"], store, selection["selected_count"],
                     settings.lower_fraction, settings.upper_fraction)

    epochs = selection["final_training_epochs"]
    model, history, _ = fit_head(train, bags, settings, settings.seed + 10000, device, epochs=epochs)
    print(f"Trained on {len(train)} patients for {epochs} epochs ({selection['selected_candidate']} slices).")

    scores = predict_proba(model, validation, bags, device, settings.batch_size)
    threshold, _ = choose_threshold(validation["label"], scores)
    metrics = summarize(validation["label"], scores, threshold)

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    history.to_csv(results / "final_training_history.csv", index=False)
    validation[["patient_id", "label"]].assign(score=scores, predicted=(scores >= threshold).astype(int)).to_csv(
        results / "validation_predictions.csv", index=False)
    write_json(results / "metrics_validation.json", metrics)

    extraction_config, encoder_sha = extraction_info(args.features)
    known = ExtractionConfig.__dataclass_fields__
    model_dir = Path(args.model_dir)
    save_head(model_dir / "head.pt", model, settings)
    write_json(model_dir / "config.json", {
        "extraction_config": {k: v for k, v in extraction_config.items() if k in known},
        "encoder_sha256": encoder_sha,
        "feature_kind": settings.feature_kind,
        "selected_count": selection["selected_count"],
        "slice_range": [settings.lower_fraction, settings.upper_fraction],
        "threshold": threshold,
        "threshold_objective": "balanced accuracy on the validation split",
        "class_names": CLASS_NAMES,
        "training_epochs": epochs,
        "train_mutant_fraction": float(train["label"].mean()),
    })
    print(f"Threshold {threshold:.3f} (validation AUROC {metrics['auroc']:.3f}, balanced accuracy "
          f"{metrics['balanced_accuracy']:.3f}). Saved {model_dir}/. Next: python evaluate.py")


if __name__ == "__main__":
    main()
