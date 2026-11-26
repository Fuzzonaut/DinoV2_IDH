"""Figures used in the README and the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

WILDTYPE_COLOR, MUTANT_COLOR = "#295d86", "#cf734b"


def plot_slice_search(results, selected_candidate, save_path=None):
    frame = results.assign(_order=results["requested_count"].astype(float).fillna(np.inf)).sort_values("_order")
    labels = ["All" if pd.isna(k) else str(int(k)) for k in frame["requested_count"]]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.errorbar(x, frame["mean_cv_auroc"], yerr=frame["std_fold_auroc"], fmt="o-", capsize=4, color=WILDTYPE_COLOR)
    winner = int(np.flatnonzero(frame["candidate"].to_numpy() == selected_candidate)[0])
    ax.scatter(x[winner], frame["mean_cv_auroc"].iloc[winner], color=MUTANT_COLOR, s=90, zorder=5, label="Selected")
    ax.set(xticks=x, xticklabels=labels, xlabel="Slices per patient", ylabel="Mean CV AUROC",
           title="Slice-count search on the training patients (error bars: fold SD)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_test_results(labels, curves, threshold, predicted, save_path=None):
    """`curves`: {name: scores}. Left: confusion matrix of `predicted`; right: ROC curves."""
    from sklearn.metrics import confusion_matrix, roc_auc_score

    cm = confusion_matrix(labels, predicted, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Wildtype", "Mutant"], yticklabels=["Wildtype", "Mutant"],
                xlabel="Predicted", ylabel="Actual", title=f"Test confusion matrix (threshold {threshold:.3f})")
    for name, scores in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        axes[1].plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(labels, scores):.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC on the test patients")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pr_and_reliability(labels, scores, bins, save_path=None):
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(recall, precision, label=f"AP = {average_precision_score(labels, scores):.3f}")
    axes[0].axhline(np.mean(labels), ls="--", color="gray", label="Mutant fraction")
    axes[0].set(xlabel="Recall (mutant)", ylabel="Precision (mutant)", title="Precision-recall", xlim=(0, 1), ylim=(0, 1.03))
    axes[0].legend()
    filled = bins[bins["patients"] > 0]
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].scatter(filled["mean_score"], filled["observed_mutant_fraction"], s=25 + 3 * filled["patients"],
                    color=WILDTYPE_COLOR)
    for row in filled.to_dict("records"):
        axes[1].annotate(f"n={row['patients']}", (row["mean_score"], row["observed_mutant_fraction"]),
                         xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[1].set(xlabel="Mean model score", ylabel="Observed mutant fraction", title="Score reliability (5 bins)",
                xlim=(0, 1), ylim=(-0.03, 1.03))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_preprocessing(prepared, image_batch_fn, modalities, title="", save_path=None):
    """Middle slice: normalised volume (left) vs. the actual 224 x 224 encoder input (right)."""
    position = len(prepared["indices"]) // 2
    z = int(prepared["indices"][position])
    processed = image_batch_fn(prepared, [position])[0].numpy()
    fig, axes = plt.subplots(len(modalities), 2, figsize=(7, 3.2 * len(modalities)))
    for i, modality in enumerate(modalities):
        axes[i, 0].imshow(prepared["arrays"][i][:, :, z].T, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[i, 1].imshow(processed[i], cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[i, 0].set_title(f"{modality}: normalised, z={z}")
        axes[i, 1].set_title(f"{modality}: encoder input")
        for ax in axes[i]:
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig
