"""Choosing the number of slices per patient by patient-level cross-validation on the TRAIN split.

For every candidate count k the same folds and seeds are used:
    outer stratified K-fold over the 345 training patients
      -> development part is split again into fit (80%) / early-stop (20%) patients
      -> the outer held-out patients score the candidate (AUROC).
The validation and test splits are never touched here.

Selection rule: highest mean CV AUROC; candidates within `near_tie_auc` (0.005) of the best
count as a tie and the smallest k wins. The final model trains for the median of the
winning candidate's early-stopping epochs.
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .features import make_bags
from .training import fit_head, predict_proba


def count_name(count):
    return "all" if count is None else f"k{int(count):03d}"


def make_cv_folds(train, settings):
    """-> list of (repeat, fold, fit, early_stop, score) DataFrames. Repeat 0 equals the original run."""
    folds = []
    for repeat in range(settings.cv_repeats):
        offset = 1000 * repeat
        splitter = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed + offset)
        for fold, (dev_idx, score_idx) in enumerate(splitter.split(train, train["label"]), 1):
            fit_idx, stop_idx = train_test_split(
                dev_idx, test_size=settings.inner_stop_fraction, stratify=train.iloc[dev_idx]["label"],
                random_state=settings.seed + offset + 100 * fold)
            parts = [train.iloc[idx].sort_values("patient_id").reset_index(drop=True)
                     for idx in (fit_idx, stop_idx, score_idx)]
            folds.append((repeat, fold, *parts))
    return folds


def evaluate_count(count, train, store, folds, settings, device):
    """Cross-validate one candidate slice count. Returns (summary row, per-fold rows, out-of-fold predictions)."""
    bags = make_bags(train["patient_id"], store, count, settings.lower_fraction, settings.upper_fraction)
    actual = np.array([len(b) for b in bags.values()])
    fold_rows, oof = [], []
    for repeat, fold, fit, early_stop, score in folds:
        seed = settings.seed + 1000 * repeat + fold
        model, _, best_epoch = fit_head(fit, bags, settings, seed, device, stop=early_stop)
        proba = predict_proba(model, score, bags, device, settings.batch_size)
        auc = float(roc_auc_score(score["label"], proba))
        fold_rows.append({"candidate": count_name(count), "repeat": repeat, "fold": fold,
                          "score_auroc": auc, "best_epoch": best_epoch})
        oof.append(score[["patient_id", "label"]].assign(candidate=count_name(count), repeat=repeat,
                                                          fold=fold, score=proba))
    aucs = [r["score_auroc"] for r in fold_rows]
    epochs = [r["best_epoch"] for r in fold_rows]
    summary = {
        "candidate": count_name(count), "requested_count": count,
        "mean_cv_auroc": float(np.mean(aucs)), "std_fold_auroc": float(np.std(aucs, ddof=1)),
        "mean_actual_slices": float(actual.mean()), "min_actual_slices": int(actual.min()),
        "max_actual_slices": int(actual.max()),
        "median_best_epoch": max(1, int(np.floor(np.median(epochs) + 0.5))),
    }
    return summary, fold_rows, pd.concat(oof, ignore_index=True)


def refinement_counts(rows, initial):
    """Up to four extra counts around the best numeric candidate (e.g. 48 and 62 around 64)."""
    numeric = sorted(k for k in initial if k is not None)
    if len(numeric) < 2:
        return []
    best = max((r for r in rows if r["requested_count"] is not None),
               key=lambda r: (r["mean_cv_auroc"], -r["requested_count"]))["requested_count"]
    i = numeric.index(best)
    left, right = numeric[max(0, i - 1)], numeric[min(len(numeric) - 1, i + 1)]
    proposals = {best - 2, best + 2, (left + best) // 2, (best + right) // 2}
    return sorted(k for k in proposals if numeric[0] <= k <= numeric[-1] and k not in initial)


def select_count(rows, tolerance):
    best_auc = max(r["mean_cv_auroc"] for r in rows)
    ties = [r for r in rows if r["mean_cv_auroc"] >= best_auc - tolerance - 1e-12]
    return min(ties, key=lambda r: (math.inf if r["requested_count"] is None else r["requested_count"],
                                    -r["mean_cv_auroc"]))


def search_slice_counts(train, store, settings, device, log=print):
    """Run the full search. Returns (selection dict, results, per-fold results, out-of-fold predictions)."""
    if set(train["split"]) != {"train"}:
        raise ValueError("The slice-count search must only see TRAIN patients.")
    folds = make_cv_folds(train, settings)
    rows, fold_rows, oof = [], [], []

    def run(count, stage):
        summary, per_fold, predictions = evaluate_count(count, train, store, folds, settings, device)
        rows.append({**summary, "stage": stage})
        fold_rows.extend(per_fold)
        oof.append(predictions)
        log(f"{summary['candidate']:>5} ({stage}): CV AUROC {summary['mean_cv_auroc']:.4f} "
            f"(fold SD {summary['std_fold_auroc']:.4f})")

    for count in settings.counts:
        run(count, "initial")
    if settings.refine:
        for count in refinement_counts(rows, settings.counts):
            run(count, "refinement")

    chosen = select_count(rows, settings.near_tie_auc)
    selection = {
        "selected_count": chosen["requested_count"], "selected_candidate": chosen["candidate"],
        "selected_cv_auroc": chosen["mean_cv_auroc"],
        "best_tested_cv_auroc": max(r["mean_cv_auroc"] for r in rows),
        "final_training_epochs": chosen["median_best_epoch"],
    }
    return selection, pd.DataFrame(rows), pd.DataFrame(fold_rows), pd.concat(oof, ignore_index=True)
