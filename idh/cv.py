"""v2 evaluation protocol: repeated stratified K-fold CV on the development patients.

Why it replaces v1's protocol
* v1 selected on 3 folds of 345 patients with a 20% inner early-stopping split (~55 patients,
  ~11 mutants per stop set), then picked the threshold on 16 validation mutants. Fold SDs were
  0.02-0.11 AUROC and the threshold did not transfer.
* v2 merges train + validation into one development set (420 patients, 87 mutants), runs
  `repeats` x `folds` CV with the SAME folds for every model, trains for a fixed number of epochs
  (no inner split), picks the threshold on all out-of-fold predictions, and predicts the test set
  with the average of all fold models (an ensemble, so no refit is needed).
The test split (75 patients) is only scored at the very end.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import select_slices
from .model import GatedAttentionMIL
from .training import collate_bags
from .utils import seed_everything


def make_folds(dev, folds, repeats, seed):
    """-> list of (repeat, fold, train_index, heldout_index) over the rows of `dev`."""
    out = []
    for r in range(repeats):
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed + 1000 * r)
        for k, (tr, ho) in enumerate(splitter.split(dev, dev["label"]), 1):
            out.append((r, k, tr, ho))
    return out


def run_cv(dev, test, folds, fit_predict, name):
    """Generic driver. `fit_predict(train_frame, [heldout_frame, test_frame], seed)` must return
    {config: (heldout_scores, test_scores)}; several configs (e.g. epochs) can share one fit.
    Returns (oof DataFrame, {config: mean test scores})."""
    oof, test_sum, n = [], {}, 0
    for r, k, tr, ho in folds:
        train, held = dev.iloc[tr].reset_index(drop=True), dev.iloc[ho].reset_index(drop=True)
        results = fit_predict(train, [held, test], seed=1000 * r + k)
        for config, (held_scores, test_scores) in results.items():
            oof.append(held[["patient_id", "label"]].assign(model=name, config=str(config), repeat=r, fold=k,
                                                            score=held_scores))
            test_sum[str(config)] = test_sum.get(str(config), 0) + np.asarray(test_scores)
        n += 1
    return pd.concat(oof, ignore_index=True), {c: s / n for c, s in test_sum.items()}


def cv_summary(oof):
    """Per (model, config): AUROC per repeat (pooled over its folds) and per fold."""
    rows = []
    for (model, config), g in oof.groupby(["model", "config"], sort=False):
        per_repeat = [roc_auc_score(x["label"], x["score"]) for _, x in g.groupby("repeat")]
        per_fold = [roc_auc_score(x["label"], x["score"]) for _, x in g.groupby(["repeat", "fold"])]
        rows.append({"model": model, "config": config, "cv_auroc": float(np.mean(per_repeat)),
                     "cv_auroc_sd_repeats": float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else 0.0,
                     "fold_auroc_sd": float(np.std(per_fold, ddof=1)), "n_folds": len(per_fold)})
    return pd.DataFrame(rows)


def mean_oof(oof):
    """One score per patient: the mean over repeats."""
    return oof.groupby("patient_id", sort=False).agg(label=("label", "first"), score=("score", "mean")).reset_index()


# --------------------------------------------------------------------------- #
# Attention-MIL head on cached slice features
# --------------------------------------------------------------------------- #
def mil_fit_predict(store, count, cfg, device, checkpoints, lower=0.0, upper=1.0):
    """Returns a fit_predict function for run_cv. Configs are the epoch checkpoints."""

    def eval_bag(pid):
        vectors, fractions, _ = store[pid]
        return vectors[select_slices(fractions, count, lower, upper)]

    def fit_predict(train, eval_frames, seed):
        seed_everything(cfg.seed + seed)
        rng = np.random.default_rng(cfg.seed + seed)
        dim = next(iter(store.values()))[0].shape[1]
        model = GatedAttentionMIL(dim, cfg.attention_dim, cfg.dropout).to(device)
        positives = int(train["label"].sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positives) / positives, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        ids, labels = train["patient_id"].tolist(), train["label"].to_numpy(dtype=np.float32)
        eligible = {pid: np.flatnonzero((store[pid][1] >= lower) & (store[pid][1] <= upper)) for pid in ids}
        eval_bags = [{pid: torch.from_numpy(np.ascontiguousarray(eval_bag(pid), dtype=np.float32))
                      for pid in f["patient_id"]} for f in eval_frames]

        def train_bag(pid):
            idx = eligible[pid]
            if cfg.bag_sampling:  # random subset: count slices, or 50-100% of all slices for count=None
                k = len(idx) if count is None else min(count, len(idx))
                if count is None:
                    k = max(1, int(rng.integers(len(idx) // 2, len(idx) + 1)))
                idx = np.sort(rng.choice(idx, size=k, replace=False))
            else:
                idx = select_slices(store[pid][1], count, lower, upper)
            return torch.from_numpy(np.ascontiguousarray(store[pid][0][idx], dtype=np.float32))

        out = {}
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            order = rng.permutation(len(ids))
            for start in range(0, len(order), cfg.batch_size):
                batch = [(train_bag(ids[i]), float(labels[i])) for i in order[start:start + cfg.batch_size]]
                x, mask, y = collate_bags(batch)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x.to(device), mask.to(device)), y.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch in checkpoints:
                model.eval()
                scores = []
                with torch.inference_mode():
                    for frame, bags in zip(eval_frames, eval_bags):
                        s = []
                        for start in range(0, len(frame), 32):
                            batch = [(bags[p], 0.0) for p in frame["patient_id"].iloc[start:start + 32]]
                            x, mask, _ = collate_bags(batch)
                            s.append(model(x.to(device), mask.to(device)).sigmoid().cpu().numpy())
                        scores.append(np.concatenate(s))
                out[f"k={'all' if count is None else count},ep={epoch}"] = tuple(scores)
        return out

    return fit_predict


# --------------------------------------------------------------------------- #
# Tabular models (tumour descriptors, age)
# --------------------------------------------------------------------------- #
def tabular_fit_predict(table, columns, kind, seed=2026):
    """`table`: DataFrame indexed by patient_id. kind: 'logreg' (L2, C tuned by inner CV) or 'gbm'."""

    def fit_predict(train, eval_frames, seed=seed):
        x = table.loc[train["patient_id"], columns].to_numpy(dtype=float)
        y = train["label"].to_numpy()
        if kind == "logreg":
            model = make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=np.logspace(-3, 1, 9), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                scoring="roc_auc", class_weight="balanced", max_iter=5000))
        elif kind == "gbm":
            model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                   l2_regularization=1.0, class_weight="balanced",
                                                   random_state=seed)
        elif kind == "plain_logreg":
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        else:
            raise ValueError(kind)
        model.fit(x, y)
        scores = [model.predict_proba(table.loc[f["patient_id"], columns].to_numpy(dtype=float))[:, 1]
                  for f in eval_frames]
        return {kind: tuple(scores)}

    return fit_predict
