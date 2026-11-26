"""v2 step C: compare all models with repeated CV on the development patients, then score the test set once.

Usage:
    python run_cv.py --features features_tumour_pred --baseline-features features

Models (all on the same 3 x 5 folds of the 420 development patients):
  * Whole-brain MIL      v1's design (DINOv2 on whole-brain slices), new CV protocol   [--baseline-features]
  * Tumour MIL           DINOv2 on tumour-centred crops + gated attention
  * Tumour descriptors   ~40 mask-derived descriptors, logistic regression and gradient boosting
  * Tumour MIL + descr.  average of the two predicted probabilities (no extra fitting)
  * Age only             reference
For the MIL models the slice count and the number of epochs are chosen by mean CV AUROC.
The threshold is chosen on the pooled out-of-fold predictions; test = mean of the 15 fold models.
"""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from idh.config import CVConfig
from idh.cv import cv_summary, make_folds, mean_oof, mil_fit_predict, run_cv, tabular_fit_predict
from idh.features import load_features, load_manifest
from idh.metrics import bootstrap_auroc_ci, choose_threshold, summarize
from idh.utils import get_device, write_json

CHECKPOINTS = (10, 20, 30, 40)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default="features_tumour_pred")
    p.add_argument("--baseline-features", default=None, help="v1 whole-brain feature cache (optional)")
    p.add_argument("--results-dir", default="results/v2")
    p.add_argument("--folds", type=int, default=CVConfig.folds)
    p.add_argument("--repeats", type=int, default=CVConfig.repeats)
    p.add_argument("--tag", default="", help="suffix for output files, e.g. _gt")
    args = p.parse_args()

    cfg = replace(CVConfig(), folds=args.folds, repeats=args.repeats)
    device = get_device()
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.features)
    dev = manifest[manifest["split"] != "test"].reset_index(drop=True)
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    folds = make_folds(dev, cfg.folds, cfg.repeats, cfg.seed)
    print(f"Development {len(dev)} ({int(dev['label'].sum())} mutant), test {len(test)} "
          f"({int(test['label'].sum())} mutant); {cfg.repeats} x {cfg.folds}-fold CV")

    oofs, tests = [], {}

    def evaluate(name, fit_predict):
        oof, test_scores = run_cv(dev, test, folds, fit_predict, name)
        oofs.append(oof)
        for config, scores in test_scores.items():
            tests[(name, config)] = scores
        summary = cv_summary(oof).sort_values("cv_auroc", ascending=False)
        best = summary.iloc[0]
        print(f"  {name:<28} best {best['config']:<14} CV AUROC {best['cv_auroc']:.3f} "
              f"(SD over repeats {best['cv_auroc_sd_repeats']:.3f}, fold SD {best['fold_auroc_sd']:.3f})")

    # --- image models
    tumour_store = load_features(args.features, manifest)
    for count in cfg.counts:
        evaluate("Tumour MIL", mil_fit_predict(tumour_store, count, cfg, device, CHECKPOINTS))
    if args.baseline_features:
        brain_manifest = load_manifest(args.baseline_features)
        brain_store = load_features(args.baseline_features, brain_manifest)
        for count in (16, 32, 64, None):
            evaluate("Whole-brain MIL", mil_fit_predict(brain_store, count, cfg, device, CHECKPOINTS, 0.05, 0.95))

    # --- tabular models
    descriptors = pd.read_csv(Path(args.features) / "descriptors.csv", dtype={"patient_id": str}).set_index("patient_id")
    columns = list(descriptors.columns)
    evaluate("Tumour descriptors", tabular_fit_predict(descriptors, columns, "logreg", cfg.seed))
    evaluate("Tumour descriptors", tabular_fit_predict(descriptors, columns, "gbm", cfg.seed))
    ages = manifest.set_index("patient_id")[["age"]]
    evaluate("Age only", tabular_fit_predict(ages, ["age"], "plain_logreg", cfg.seed))

    oof = pd.concat(oofs, ignore_index=True)
    summary = cv_summary(oof)

    # --- choose each model's best config, build the combination
    best = summary.sort_values("cv_auroc", ascending=False).groupby("model", sort=False).head(1)
    chosen = {r.model: r.config for r in best.itertuples()}
    selected = oof[[chosen.get(m) == c for m, c in zip(oof["model"], oof["config"])]]
    mil = selected[selected["model"] == "Tumour MIL"]
    desc = selected[selected["model"] == "Tumour descriptors"]
    keys = ["patient_id", "label", "repeat", "fold"]
    combo = mil[keys + ["score"]].merge(desc[keys + ["score"]], on=keys, suffixes=("_a", "_b"))
    combo = combo.assign(model="Tumour MIL + descriptors", config=f"{chosen['Tumour MIL']} + {chosen['Tumour descriptors']}",
                         score=(combo["score_a"] + combo["score_b"]) / 2)[keys + ["model", "config", "score"]]
    tests[("Tumour MIL + descriptors", combo["config"].iloc[0])] = (
        tests[("Tumour MIL", chosen["Tumour MIL"])] + tests[("Tumour descriptors", chosen["Tumour descriptors"])]) / 2
    selected = pd.concat([selected, combo], ignore_index=True)
    final = cv_summary(selected).sort_values("cv_auroc", ascending=False).reset_index(drop=True)

    # --- thresholds from OOF, then the test set (once)
    rows, test_preds, labels = [], test[["patient_id", "label"]].copy(), test["label"].to_numpy()
    for r in final.itertuples():
        pooled = mean_oof(selected[(selected["model"] == r.model) & (selected["config"] == r.config)])
        threshold, _ = choose_threshold(pooled["label"], pooled["score"])
        scores = tests[(r.model, r.config)]
        test_preds[r.model] = scores
        m = summarize(labels, scores, threshold)
        rows.append({"model": r.model, "config": r.config, "cv_auroc": r.cv_auroc,
                     "cv_auroc_sd_repeats": r.cv_auroc_sd_repeats, "fold_auroc_sd": r.fold_auroc_sd,
                     "oof_auroc_ci": bootstrap_auroc_ci(pooled["label"], pooled["score"], 1000, cfg.seed),
                     "test_auroc": m["auroc"],
                     "test_auroc_ci": bootstrap_auroc_ci(labels, scores, cfg.bootstrap_repeats, cfg.seed + 50000),
                     "threshold": threshold, "test_sensitivity": m["sensitivity"], "test_specificity": m["specificity"],
                     "test_balanced_accuracy": m["balanced_accuracy"], "test_accuracy": m["accuracy"],
                     "test_tp": m["tp"], "test_fn": m["fn"], "test_tn": m["tn"], "test_fp": m["fp"]})
    table = pd.DataFrame(rows)

    tag = args.tag
    summary.to_csv(out / f"cv_all_configs{tag}.csv", index=False)
    table.to_csv(out / f"model_comparison{tag}.csv", index=False)
    oof.to_csv(out / f"oof_predictions{tag}.csv", index=False)
    test_preds.to_csv(out / f"test_predictions{tag}.csv", index=False)
    write_json(out / f"model_comparison{tag}.json", {"cv_config": asdict(cfg), "checkpoints": CHECKPOINTS,
                                                    "features": str(args.features), "models": rows})
    plot(table, labels, test_preds, out / f"model_comparison{tag}.png")

    show = table[["model", "config", "cv_auroc", "cv_auroc_sd_repeats", "test_auroc", "test_sensitivity",
                  "test_specificity"]]
    print(show.round(3).to_string(index=False))
    print(f"Saved to {out}/")


def plot(table, labels, test_preds, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    t = table.iloc[::-1]
    axes[0].barh(t["model"], t["cv_auroc"], xerr=t["cv_auroc_sd_repeats"], color="#295d86", alpha=0.85, label="CV (dev)")
    axes[0].scatter(t["test_auroc"], t["model"], color="#cf734b", zorder=5, label="Test")
    axes[0].set(xlim=(0.5, 1.0), xlabel="AUROC", title="CV (3 x 5-fold, 420 patients) vs. test (75)")
    axes[0].legend(loc="lower right")
    for model in table["model"]:
        fpr, tpr, _ = roc_curve(labels, test_preds[model])
        axes[1].plot(fpr, tpr, label=f"{model} ({roc_auc_score(labels, test_preds[model]):.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC")
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
