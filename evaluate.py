"""Step 5: evaluate the locked model once on the held-out test patients.

Usage:
    python evaluate.py --features features --model-dir model

Compares the image model with two non-imaging references:
  * training prevalence: always predicts "wildtype" (what plain accuracy has to beat);
  * age only: logistic regression on age at MRI (fit on train, threshold chosen on validation).
Writes results/metrics_test.json, test_metrics.csv, test_predictions.csv, predictive_values_test.csv,
reliability_test.csv, test_results.png and pr_reliability_test.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from idh.config import SearchConfig
from idh.features import load_features, load_manifest, make_bags
from idh.metrics import (age_baseline_scores, bootstrap_auroc_ci, choose_threshold, proportion_table,
                         reliability_bins, summarize)
from idh.plots import plot_pr_and_reliability, plot_test_results
from idh.training import load_head, predict_proba
from idh.utils import get_device, read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="features")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--bootstrap", type=int, default=SearchConfig.bootstrap_repeats)
    args = p.parse_args()

    seed = SearchConfig.seed
    device = get_device()
    info = read_json(Path(args.model_dir) / "config.json")
    head = load_head(Path(args.model_dir) / "head.pt", device)
    lower, upper = info["slice_range"]

    manifest = load_manifest(args.features)
    parts = {name: manifest[manifest["split"] == name].reset_index(drop=True)
             for name in ("train", "validation", "test")}
    test = parts["test"]
    store = load_features(args.features, test, info["feature_kind"])
    bags = make_bags(test["patient_id"], store, info["selected_count"], lower, upper)
    scores = predict_proba(head, test, bags, device)
    labels = test["label"].to_numpy()

    # Image model at the threshold locked on validation.
    rows = {"DINOv2 + attention MIL": summarize(labels, scores, info["threshold"])}
    rows["DINOv2 + attention MIL"]["auroc_ci"] = bootstrap_auroc_ci(labels, scores, args.bootstrap, seed + 50000)

    # Reference 1: constant score = training mutant fraction (< 0.5, so always "wildtype").
    prevalence = np.full(len(test), info.get("train_mutant_fraction", parts["train"]["label"].mean()))
    rows["Always wildtype (prevalence)"] = summarize(labels, prevalence, 0.5)

    # Reference 2: age only. Same protocol as the image model: fit on train, threshold on validation.
    age_val, _ = age_baseline_scores(parts["train"], parts["validation"])
    age_threshold, _ = choose_threshold(parts["validation"]["label"], age_val)
    age_test, _ = age_baseline_scores(parts["train"], test)
    rows["Age only (logistic regression)"] = summarize(labels, age_test, age_threshold)
    rows["Age only (logistic regression)"]["auroc_ci"] = bootstrap_auroc_ci(labels, age_test, args.bootstrap,
                                                                            seed + 50000)

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics_test.json", rows)
    table = pd.DataFrame(rows).T
    table.to_csv(out / "test_metrics.csv")
    predicted = (scores >= info["threshold"]).astype(int)
    test[["patient_id", "label"]].assign(score=scores, predicted=predicted, age_score=age_test).to_csv(
        out / "test_predictions.csv", index=False)
    proportion_table(rows["DINOv2 + attention MIL"]).to_csv(out / "predictive_values_test.csv", index=False)
    bins = reliability_bins(labels, scores)
    bins.to_csv(out / "reliability_test.csv", index=False)

    plot_test_results(labels, {"DINOv2 + MIL": scores, "Age only": age_test}, info["threshold"], predicted,
                      out / "test_results.png")
    plot_pr_and_reliability(labels, scores, bins, out / "pr_reliability_test.png")

    cols = ["auroc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "threshold"]
    print(table[cols].astype(float).round(3).to_string())
    print(f"Saved to {out}/")


if __name__ == "__main__":
    main()
