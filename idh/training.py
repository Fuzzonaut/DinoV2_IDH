"""Fitting the attention head on cached slice features."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from .model import GatedAttentionMIL
from .utils import seed_everything


class FeatureBags(Dataset):
    def __init__(self, frame, bags):
        self.rows = frame[["patient_id", "label"]].to_dict("records")
        self.bags = bags

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return torch.from_numpy(self.bags[row["patient_id"]]), float(row["label"])


def collate_bags(batch):
    """Pad bags of different lengths to (B, S_max, D) and return a mask of real slices."""
    dimension = batch[0][0].shape[1]
    max_slices = max(len(x) for x, _ in batch)
    values = torch.zeros(len(batch), max_slices, dimension, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_slices, dtype=torch.bool)
    for i, (features, _) in enumerate(batch):
        values[i, :len(features)] = features
        mask[i, :len(features)] = True
    labels = torch.tensor([y for _, y in batch], dtype=torch.float32)
    return values, mask, labels


def bag_loader(frame, bags, batch_size, shuffle=False, seed=2026):
    return DataLoader(FeatureBags(frame, bags), batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_bags, generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def predict_proba(model, frame, bags, device, batch_size=16):
    model.eval()
    outputs = [model(x.to(device), m.to(device)).sigmoid().cpu().numpy()
               for x, m, _ in bag_loader(frame, bags, batch_size)]
    return np.concatenate(outputs)


def fit_head(train, bags, settings, seed, device, stop=None, epochs=None):
    """Train a fresh head.

    * With `stop` (a separate set of patients): early stopping on its AUROC (patience
      `settings.patience`, at most `settings.max_epochs`), the best epoch's weights are returned.
    * Without `stop`: train for exactly `epochs` epochs (used for the final model).

    Returns (model, history DataFrame, best_epoch).
    """
    if (stop is None) == (epochs is None):
        raise ValueError("Pass either `stop` (early stopping) or `epochs` (fixed length).")
    seed_everything(seed)
    dimension = next(iter(bags.values())).shape[1]
    model = GatedAttentionMIL(dimension, settings.attention_dim, settings.dropout).to(device)

    # Class weighting from the fitting patients only (~4 wildtype per mutant).
    positives = int(train["label"].sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positives) / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loader = bag_loader(train, bags, settings.batch_size, shuffle=True, seed=seed)

    history, best_auc, best_epoch, best_state, waiting = [], -np.inf, 0, None, 0
    for epoch in range(1, (epochs or settings.max_epochs) + 1):
        model.train()
        total = 0.0
        for features, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features.to(device), mask.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(labels)

        stop_auc = None
        if stop is not None:
            stop_auc = float(roc_auc_score(stop["label"], predict_proba(model, stop, bags, device, settings.batch_size)))
        history.append({"epoch": epoch, "train_loss": total / len(train), "stop_auroc": stop_auc})

        if stop is None or stop_auc > best_auc + 1e-12:
            best_auc = stop_auc if stop is not None else best_auc
            best_epoch, waiting = epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waiting += 1
            if waiting >= settings.patience:
                break

    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history), best_epoch


def save_head(path, model, settings, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "dimension": int(model.feature_norm.normalized_shape[0]),
                "attention_dim": int(settings.attention_dim), "dropout": float(settings.dropout),
                **(extra or {})}, path)


def load_head(path, device="cpu"):
    """Loads checkpoints written by `save_head` and by the original Kaggle notebook (same format)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GatedAttentionMIL(ckpt["dimension"], ckpt["attention_dim"], ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
