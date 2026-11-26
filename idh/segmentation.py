"""3-D brain-tumour segmentation with a MONAI SegResNet, trained on the UCSF-PDGM expert masks.

Input : FLAIR, T1, T1c, T2 (z-scored per volume on brain voxels, cropped to the brain).
Output: the three standard BraTS regions as sigmoid channels
        TC (tumour core = necrosis + enhancing), WT (whole tumour = TC + oedema), ET (enhancing tumour),
        converted back to the dataset's label map: 1 necrosis, 2 oedema, 4 enhancing.

Cross-fitting: the development patients are split into K folds; the model of fold k is trained on the
other folds and segments fold k, so no development patient is segmented by a model that saw its mask.
Test patients are segmented by the average of the K fold models and never used for training.
"""

import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import SEG_MODALITIES

REGIONS = ("TC", "WT", "ET")


# --------------------------------------------------------------------------- #
# Volumes
# --------------------------------------------------------------------------- #
def load_canonical(path, dtype=np.float32):
    image = nib.as_closest_canonical(nib.load(str(path)))
    return image, np.asarray(image.dataobj).astype(dtype)


def load_case(row, with_label=True, margin=4):
    """-> dict(image (4, x, y, z) float16 z-scored, label (x, y, z) uint8 or None, bbox, shape, affine)."""
    arrays, reference = [], None
    for modality in SEG_MODALITIES:
        image, array = load_canonical(row[modality])
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-3):
            raise ValueError(f"{modality} is not on the FLAIR grid: {row[modality]}")
        arrays.append(array)
    brain = np.logical_or.reduce([np.abs(a) > 1e-8 for a in arrays])
    nonzero = [np.flatnonzero(brain.any(axis=tuple(j for j in range(3) if j != ax))) for ax in range(3)]
    bbox = [(max(0, int(v[0]) - margin), min(brain.shape[i], int(v[-1]) + 1 + margin)) for i, v in enumerate(nonzero)]
    crop = tuple(slice(a, b) for a, b in bbox)

    channels = []
    for array in arrays:
        values = array[brain]
        normalized = np.zeros_like(array)
        normalized[brain] = (values - values.mean()) / (values.std() + 1e-6)
        channels.append(normalized[crop])
    label = None
    if with_label:
        _, label = load_canonical(row["seg"], np.uint8)
        label = label[crop]
    return {"image": np.stack(channels).astype(np.float16), "label": label, "bbox": np.array(bbox),
            "shape": np.array(reference.shape), "affine": reference.affine}


def brats_to_regions(label):
    """Label map (0/1/2/4) -> (3, ...) float masks TC, WT, ET."""
    return np.stack([np.isin(label, (1, 4)), label > 0, label == 4]).astype(np.float32)


def regions_to_brats(regions):
    """(3, ...) boolean TC, WT, ET -> label map 0/1/2/4 (nested: ET inside TC inside WT)."""
    tc, wt, et = regions
    out = np.zeros(wt.shape, dtype=np.uint8)
    out[wt | tc | et] = 2
    out[tc | et] = 1
    out[et] = 4
    return out


def save_mask(label_cropped, case, path):
    full = np.zeros(tuple(case["shape"]), dtype=np.uint8)
    (x0, x1), (y0, y1), (z0, z1) = case["bbox"]
    full[x0:x1, y0:y1, z0:z1] = label_cropped
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, case["affine"]), str(path))


def dice(prediction, truth):
    """Dice per region; 1.0 when both are empty."""
    p, t = brats_to_regions(prediction) > 0, brats_to_regions(truth) > 0
    scores = {}
    for i, name in enumerate(REGIONS):
        denominator = p[i].sum() + t[i].sum()
        scores[name] = 1.0 if denominator == 0 else float(2 * (p[i] & t[i]).sum() / denominator)
    return scores


# --------------------------------------------------------------------------- #
# Cache (decoded once, so training is GPU-bound rather than gzip-bound)
# --------------------------------------------------------------------------- #
def build_cache(cohort, cache_dir, log=print):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for i, row in enumerate(cohort.to_dict("records"), 1):
        path = cache_dir / f"{row['patient_id']}.npz"
        if path.exists():
            continue
        case = load_case(row)
        tmp = path.with_name(path.name + ".part.npz")
        np.savez_compressed(tmp, **case)  # ~15 MB per patient
        tmp.replace(path)
        if i % 50 == 0:
            log(f"  cached {i}/{len(cohort)} ({time.time() - started:.0f}s)")


def read_case(cache_dir, patient_id):
    with np.load(Path(cache_dir) / f"{patient_id}.npz") as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
class RandomCrops(Dataset):
    """One random ROI per case per epoch; part of the crops are centred on a tumour voxel. Flips + intensity jitter."""

    def __init__(self, patient_ids, cache_dir, roi, tumour_probability, seed):
        self.ids, self.cache_dir, self.roi = list(patient_ids), cache_dir, np.array(roi)
        self.tumour_probability = tumour_probability
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        case = read_case(self.cache_dir, self.ids[i])
        image, label = case["image"].astype(np.float32), brats_to_regions(case["label"])
        # pad to at least the ROI
        pad = [(0, 0)] + [(0, max(0, int(r - s))) for r, s in zip(self.roi, image.shape[1:])]
        image, label = np.pad(image, pad), np.pad(label, pad)
        shape = np.array(image.shape[1:])
        tumour = np.argwhere(label[1] > 0)
        if len(tumour) and rng.random() < self.tumour_probability:
            centre = tumour[rng.integers(len(tumour))]
        else:
            centre = rng.integers(0, shape)
        start = np.clip(centre - self.roi // 2, 0, shape - self.roi)
        window = tuple(slice(s, s + r) for s, r in zip(start, self.roi))
        image, label = image[(slice(None),) + window], label[(slice(None),) + window]
        for axis in (1, 2, 3):
            if rng.random() < 0.5:
                image, label = np.flip(image, axis), np.flip(label, axis)
        scale = 1 + rng.uniform(-0.1, 0.1, size=(4, 1, 1, 1))
        shift = rng.uniform(-0.1, 0.1, size=(4, 1, 1, 1))
        image = image * scale + shift
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), torch.from_numpy(
            np.ascontiguousarray(label))


def build_model(cfg):
    from monai.networks.nets import SegResNet

    return SegResNet(spatial_dims=3, in_channels=len(SEG_MODALITIES), out_channels=3, init_filters=cfg.init_filters,
                     blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2)


def train_model(patient_ids, cache_dir, cfg, device, seed, num_workers=4, log=print):
    from monai.losses import DiceLoss

    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    dataset = RandomCrops(patient_ids, cache_dir, cfg.roi, cfg.tumour_crop_probability, seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                        persistent_workers=False, generator=torch.Generator().manual_seed(seed))
    dice_loss = DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(1, cfg.epochs + 1):
        dataset.epoch = epoch
        model.train()
        total, started = 0.0, time.time()
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(image)
                loss = dice_loss(logits, label) + bce(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(image)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            log(f"    epoch {epoch:3d}/{cfg.epochs}  loss {total / (len(loader) * cfg.batch_size):.4f}  "
                f"({time.time() - started:.0f}s/epoch)")
    return model.eval()


@torch.inference_mode()
def predict_probabilities(models, image, cfg, device):
    """Average sigmoid output of `models` on one cropped case (4, x, y, z) -> (3, x, y, z) float32."""
    from monai.inferers import sliding_window_inference

    x = torch.from_numpy(image.astype(np.float32))[None].to(device)
    total = None
    for model in models:
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = sliding_window_inference(x, cfg.roi, sw_batch_size=2, predictor=model,
                                              overlap=cfg.sw_overlap, mode="gaussian")
        probability = torch.sigmoid(logits.float())[0]
        total = probability if total is None else total + probability
    return (total / len(models)).cpu().numpy()


def save_weights(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, path)  # fp16: half the size


def load_weights(path, cfg, device):
    model = build_model(cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.to(device).eval()
