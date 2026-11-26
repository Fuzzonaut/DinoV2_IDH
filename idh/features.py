"""Frozen DINOv2 slice encoder, the on-disk feature cache, and slice sampling.

Every nonempty axial slice of every patient is encoded once and cached, so the slice-count
search and all later steps only need the small feature files (~200 MB for 495 patients),
not the MRI volumes or a GPU.

Cache layout (compatible with the feature pack exported by the original Kaggle notebook):

    features/
    ├── extraction.json          settings, encoder checksum, feature signature
    ├── patient_manifest.csv     patient_id, label, split, age, sex, grade, n_slices, feature_file
    └── features/<patient>.npz   cls_features, patch_mean_features (n_slices x 384),
                                 slice_indices, slice_z_mm, slice_fraction, nonzero_area_fraction
"""

import contextlib
import hashlib
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .preprocessing import image_batch, prepare_volume
from .utils import read_json, write_json

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class FrozenDINOv2(nn.Module):
    """DINOv2 with all weights frozen. Input (N, 3, H, W) in [0, 1] -> (CLS token, mean patch token)."""

    def __init__(self, model_id="facebook/dinov2-small", revision=None, cache_dir=None, token=None, encoder=None):
        super().__init__()
        if encoder is None:
            from transformers import Dinov2Model

            encoder = Dinov2Model.from_pretrained(model_id, revision=revision, cache_dir=cache_dir,
                                                  use_safetensors=True, token=token)
        self.encoder = encoder.requires_grad_(False)
        self.register_buffer("input_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()

    def train(self, mode=True):  # always stays in eval mode
        return super().train(False)

    @property
    def feature_dim(self):
        return int(self.encoder.config.hidden_size)

    @torch.inference_mode()
    def forward(self, images):
        hidden = self.encoder(pixel_values=(images - self.input_mean) / self.input_std,
                              return_dict=True).last_hidden_state
        return hidden[:, 0].float(), hidden[:, 1:].float().mean(dim=1)


def load_encoder(config, device, cache_dir=None, token=None):
    return FrozenDINOv2(config.model_id, config.model_revision, cache_dir, token).to(device).eval()


def model_digest(model):
    """SHA-256 over all weights; used to check the encoder is unchanged and matches the cache."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def feature_signature(config, encoder_digest):
    """Identifies everything that determines the feature values (not the split or the head)."""
    identity = {k: v for k, v in asdict(config).items()
                if k not in {"seed", "slice_batch_size", "test_fraction", "validation_fraction"}}
    identity["encoder_sha256"] = encoder_digest
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


@torch.inference_mode()
def encode_volume(prepared, model, config, device):
    """Encode every eligible slice of one prepared patient -> (cls, patch_mean), each (n_slices, D)."""
    use_amp = config.use_mixed_precision and device.type == "cuda"
    cls_parts, patch_parts = [], []
    count = len(prepared["indices"])
    for start in range(0, count, config.slice_batch_size):
        positions = np.arange(start, min(start + config.slice_batch_size, count))
        images = image_batch(prepared, positions, config.image_size).to(device)
        context = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
        with context:
            cls, patch = model(images)
        cls_parts.append(cls.cpu().numpy().astype(np.float32))
        patch_parts.append(patch.cpu().numpy().astype(np.float32))
    return np.concatenate(cls_parts), np.concatenate(patch_parts)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def read_npz(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def extract_cohort(cohort, model, config, device, out_dir):
    """Encode all patients into `out_dir`. Already-finished patients are reused (resumable)."""
    out_dir = Path(out_dir)
    (out_dir / "features").mkdir(parents=True, exist_ok=True)
    digest = model_digest(model)
    signature = feature_signature(config, digest)
    info_path = out_dir / "extraction.json"
    if info_path.exists() and read_json(info_path)["feature_signature"] != signature:
        raise ValueError(f"{out_dir} holds features from different settings; use another --out-dir.")
    write_json(info_path, {"feature_signature": signature, "encoder_sha256": digest,
                           "feature_dim": model.feature_dim, "config": asdict(config), "complete": False})

    rows = []
    for row in tqdm(cohort.to_dict("records"), desc="DINOv2 feature extraction"):
        relative = f"features/{row['patient_id']}.npz"
        path = out_dir / relative
        n_slices = None
        if path.exists():
            try:
                n_slices = len(read_npz(path)["cls_features"])
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                n_slices = None  # corrupt / partial file -> re-extract
        if n_slices is None:
            prepared = prepare_volume(row, config.modalities, config.crop_margin_voxels)
            cls, patch = encode_volume(prepared, model, config, device)
            tmp = path.with_name(path.name + ".part")
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f, cls_features=cls, patch_mean_features=patch,
                    slice_indices=prepared["indices"], slice_z_mm=prepared["z_mm"],
                    slice_fraction=prepared["fractions"], nonzero_area_fraction=prepared["area"])
            tmp.replace(path)
            n_slices = len(cls)
        rows.append({"patient_id": row["patient_id"], "n_slices": n_slices, "feature_file": relative})

    if model_digest(model) != digest:
        raise RuntimeError("Encoder weights changed during extraction.")
    keep = [c for c in ("patient_id", "label", "split", "age", "sex", "grade") if c in cohort.columns]
    manifest = cohort[keep].merge(pd.DataFrame(rows), on="patient_id", validate="one_to_one")
    manifest.to_csv(out_dir / "patient_manifest.csv", index=False)
    write_json(info_path, {**read_json(info_path), "complete": True,
                           "patients": len(manifest), "slices": int(manifest["n_slices"].sum())})
    return manifest


def load_manifest(feature_dir):
    feature_dir = Path(feature_dir)
    manifest = pd.read_csv(feature_dir / "patient_manifest.csv", dtype={"patient_id": str})
    if manifest["patient_id"].duplicated().any() or set(manifest["label"]) != {0, 1}:
        raise ValueError("The feature manifest needs one row per patient and both classes.")
    return manifest.sort_values("patient_id").reset_index(drop=True)


def load_features(feature_dir, manifest, feature_kind="cls"):
    """-> {patient_id: (vectors (n_slices, D), slice_fraction (n_slices,), slice_z_mm (n_slices,))}."""
    store = {}
    for record in tqdm(manifest.to_dict("records"), desc="Loading features", leave=False):
        data = read_npz(Path(feature_dir) / record["feature_file"])
        store[record["patient_id"]] = (select_feature_kind(data, feature_kind),
                                       data["slice_fraction"], data["slice_z_mm"])
    return store


def select_feature_kind(data, feature_kind):
    if feature_kind == "cls":
        return data["cls_features"]
    if feature_kind == "patch_mean":
        return data["patch_mean_features"]
    if feature_kind == "concat":
        return np.concatenate([data["cls_features"], data["patch_mean_features"]], axis=1)
    raise ValueError("feature_kind must be cls, patch_mean or concat.")


# --------------------------------------------------------------------------- #
# Slice sampling
# --------------------------------------------------------------------------- #
def select_slices(fractions, count=None, lower=0.05, upper=0.95):
    """Positions of `count` evenly spaced slices within the [lower, upper] part of the brain extent.

    Uses only slice positions (no labels, no tumour masks). If fewer than `count` slices are
    eligible, or count is None ("all"), every eligible slice is returned.
    """
    eligible = np.flatnonzero((fractions >= lower) & (fractions <= upper))
    if not len(eligible):
        raise ValueError("No slices in the requested position interval.")
    if count is None or count >= len(eligible):
        return eligible
    locations = fractions[eligible]
    targets = (np.asarray([(locations[0] + locations[-1]) / 2]) if count == 1
               else np.linspace(locations[0], locations[-1], count))
    remaining, chosen = list(range(len(eligible))), []
    for target in targets:  # nearest unused slice to each target position
        position = min(remaining, key=lambda j: abs(float(locations[j]) - target))
        chosen.append(position)
        remaining.remove(position)
    return eligible[np.sort(chosen)]


def make_bags(patient_ids, store, count, lower=0.05, upper=0.95):
    """-> {patient_id: (n_selected, D) float32 array}."""
    bags = {}
    for pid in patient_ids:
        vectors, fractions, _ = store[pid]
        bags[pid] = np.ascontiguousarray(vectors[select_slices(fractions, count, lower, upper)], dtype=np.float32)
    return bags
