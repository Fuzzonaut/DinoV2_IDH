"""Binary evaluation (positive class = IDH mutant)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)


def summarize(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "patients": int(len(labels)), "mutants": int(labels.sum()),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "ppv": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv": float(tn / (tn + fn)) if tn + fn else np.nan,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "log_loss": float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": float(brier_score_loss(labels, scores)),
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(labels, scores):
    """Threshold maximising balanced accuracy on a 0.05..0.95 grid; ties go to the value closest to 0.5."""
    grid = np.linspace(0.05, 0.95, 181)
    values = np.array([balanced_accuracy_score(labels, np.asarray(scores) >= t) for t in grid])
    best = grid[values >= values.max() - 1e-12]
    threshold = float(best[np.argmin(np.abs(best - 0.5))])
    return threshold, pd.DataFrame({"threshold": grid, "balanced_accuracy": values})


def bootstrap_auroc_ci(labels, scores, repeats=1000, seed=0):
    """95% percentile interval, resampling patients within each class (model and threshold fixed)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    neg, pos = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(neg, len(neg), replace=True), rng.choice(pos, len(pos), replace=True)])
        values.append(roc_auc_score(labels[idx], scores[idx]))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def proportion_table(metrics):
    """Accuracy / sensitivity / specificity / PPV / NPV with Wilson 95% intervals, from a `summarize` dict."""
    tn, fp, fn, tp = metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]
    rows = []
    for name, k, n in [("accuracy", tp + tn, tp + tn + fp + fn), ("sensitivity", tp, tp + fn),
                       ("specificity", tn, tn + fp), ("ppv", tp, tp + fp), ("npv", tn, tn + fn)]:
        low, high = wilson_interval(k, n)
        rows.append({"metric": name, "numerator": k, "denominator": n,
                     "estimate": k / n if n else np.nan, "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows)


def reliability_bins(labels, scores, n_bins=5):
    labels, scores = np.asarray(labels), np.asarray(scores)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        members = ids == b
        n, k = int(members.sum()), int(labels[members].sum())
        low, high = wilson_interval(k, n)
        rows.append({"bin_lower": edges[b], "bin_upper": edges[b + 1], "patients": n, "mutants": k,
                     "mean_score": float(scores[members].mean()) if n else np.nan,
                     "observed_mutant_fraction": k / n if n else np.nan,
                     "wilson_low": low, "wilson_high": high})
    return pd.DataFrame(rows)


def age_baseline_scores(train, test):
    """Logistic regression on age at MRI only, fit on training patients.

    IDH-mutant gliomas occur in much younger patients (training means: 40 vs 62 years), so this
    is the obvious non-imaging reference: the image model is only useful if it beats it.
    """
    model = LogisticRegression().fit(train[["age"]].astype(float), train["label"])
    return model.predict_proba(test[["age"]].astype(float))[:, 1], model
