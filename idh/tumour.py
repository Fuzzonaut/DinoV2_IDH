"""Using a tumour mask: (1) tumour-centred DINOv2 input, (2) hand-crafted tumour descriptors.

Mask labels (BraTS convention): 1 necrosis, 2 oedema, 4 enhancing tumour.
WT (whole tumour) = 1 + 2 + 4, TC (tumour core) = 1 + 4, ET = 4.
"""

import numpy as np
from scipy import ndimage

from .segmentation import load_canonical


# --------------------------------------------------------------------------- #
# Tumour-centred slices for the encoder
# --------------------------------------------------------------------------- #
def tumour_slices(prepared, mask, tcfg):
    """Turn a whole-brain `prepared` volume (see preprocessing.prepare_volume) into a tumour-focused one.

    Keeps only axial slices with tumour and crops every slice to one square box around the tumour
    (bounding box + margin, at least `min_crop_voxels` wide). The result can be fed to image_batch().
    """
    if prepared["resampled_oblique"]:
        raise ValueError("Oblique volumes are not supported for tumour cropping.")
    wt = mask > 0
    area = wt.sum(axis=(0, 1))
    indices = np.flatnonzero(area >= tcfg.min_tumour_voxels)
    if len(indices) < 2:
        indices = np.flatnonzero(area > 0)
    if len(indices) < 2:  # segmentation found (almost) nothing: fall back to the whole brain
        return prepared, False

    xs, ys = np.flatnonzero(wt.any(axis=(1, 2))), np.flatnonzero(wt.any(axis=(0, 2)))
    side = max(xs[-1] - xs[0] + 1, ys[-1] - ys[0] + 1) + 2 * tcfg.margin_voxels
    side = int(min(max(side, tcfg.min_crop_voxels), min(mask.shape[:2])))
    starts, stops = [], []
    for lo, hi, size in ((xs[0], xs[-1], mask.shape[0]), (ys[0], ys[-1], mask.shape[1])):
        start = int(np.clip((lo + hi + 1) // 2 - side // 2, 0, size - side))
        starts.append(start)
        stops.append(start + side)

    z_mm = indices * prepared["voxel_spacing"][2]
    focused = {
        **prepared,
        "indices": indices.astype(np.int32),
        "crop_starts": starts + [0], "crop_stops": stops + [mask.shape[2]],
        "z_mm": z_mm.astype(np.float64),
        "fractions": ((indices - indices[0]) / max(1, indices[-1] - indices[0])).astype(np.float32),
        "area": (area[indices] / area.max()).astype(np.float32),
    }
    return focused, True


def load_mask(path, reference_shape):
    _, mask = load_canonical(path, np.uint8)
    if mask.shape != tuple(reference_shape):
        raise ValueError(f"Mask shape {mask.shape} does not match the MRI {tuple(reference_shape)}: {path}")
    return mask


# --------------------------------------------------------------------------- #
# Hand-crafted tumour descriptors
# --------------------------------------------------------------------------- #
def _relative(values, reference):
    return values / (reference + 1e-6)


def tumour_descriptors(row, mask):
    """~40 descriptors: sub-region volumes and ratios, intensities relative to normal brain, T2-FLAIR
    mismatch, enhancement, location and shape. NaN-free (empty regions give 0 / neutral values)."""
    images = {}
    for modality in ("FLAIR", "T1", "T1c", "T2"):
        image, images[modality] = load_canonical(row[modality])
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    voxel_ml = float(np.prod(spacing)) / 1000

    brain = np.logical_or.reduce([np.abs(a) > 1e-8 for a in images.values()])
    wt, tc, et = mask > 0, np.isin(mask, (1, 4)), mask == 4
    necrosis, oedema = mask == 1, mask == 2
    normal = brain & ~ndimage.binary_dilation(wt, iterations=3)
    if normal.sum() < 1000:
        normal = brain

    f = {}
    volumes = {"wt": wt, "tc": tc, "et": et, "necrosis": necrosis, "oedema": oedema}
    for name, region in volumes.items():
        f[f"log_volume_{name}_ml"] = float(np.log1p(region.sum() * voxel_ml))
    n_wt = max(1, int(wt.sum()))
    f["frac_et_of_wt"] = et.sum() / n_wt
    f["frac_necrosis_of_wt"] = necrosis.sum() / n_wt
    f["frac_oedema_of_wt"] = oedema.sum() / n_wt
    f["frac_et_of_tc"] = et.sum() / max(1, int(tc.sum()))
    f["enhancing_present"] = float(et.sum() * voxel_ml > 0.5)

    relative = {}
    for modality, array in images.items():
        reference = np.median(array[normal])
        relative[modality] = _relative(array, reference)
        values = relative[modality][wt] if wt.any() else np.ones(1)
        p10, p50, p90 = np.percentile(values, [10, 50, 90])
        f[f"{modality}_wt_median"], f[f"{modality}_wt_p10"], f[f"{modality}_wt_p90"] = p50, p10, p90
        f[f"{modality}_wt_iqr"] = float(np.subtract(*np.percentile(values, [75, 25])))
        non_enhancing = wt & ~et
        f[f"{modality}_nonenh_median"] = float(np.median(relative[modality][non_enhancing])) if non_enhancing.any() else p50

    # T2-FLAIR mismatch: bright on T2 but relatively suppressed on FLAIR (a known IDH-mutant sign).
    if wt.any():
        t2, flair = relative["T2"][wt], relative["FLAIR"][wt]
        f["flair_t2_ratio"] = float(np.median(flair / (t2 + 1e-6)))
        f["mismatch_fraction"] = float(np.mean((t2 > np.median(t2)) & (flair < np.median(flair))))
        f["t1c_t1_ratio_median"] = float(np.median(relative["T1c"][wt] / (relative["T1"][wt] + 1e-6)))
        f["t1c_t1_ratio_p90"] = float(np.percentile(relative["T1c"][wt] / (relative["T1"][wt] + 1e-6), 90))
    else:
        f.update(flair_t2_ratio=1.0, mismatch_fraction=0.0, t1c_t1_ratio_median=1.0, t1c_t1_ratio_p90=1.0)

    # Location relative to the brain bounding box (x = left-right, y = posterior-anterior, z = inferior-superior).
    bx, by, bz = [np.flatnonzero(brain.any(axis=tuple(j for j in range(3) if j != a))) for a in range(3)]
    if wt.any():
        cx, cy, cz = np.argwhere(wt).mean(axis=0)
        f["loc_lateral"] = abs(cx - (bx[0] + bx[-1]) / 2) / ((bx[-1] - bx[0]) / 2)
        f["loc_anterior"] = (cy - by[0]) / (by[-1] - by[0])
        f["loc_superior"] = (cz - bz[0]) / (bz[-1] - bz[0])
    else:
        f.update(loc_lateral=0.0, loc_anterior=0.5, loc_superior=0.5)

    # Shape.
    if wt.sum() >= 10:
        coords = np.argwhere(wt) * spacing
        eig = np.sort(np.linalg.eigvalsh(np.cov(coords.T)))[::-1]
        f["elongation"] = float(np.sqrt(eig[0] / max(eig[2], 1e-6)))
        surface = (wt & ~ndimage.binary_erosion(wt)).sum() * float(np.mean(spacing) ** 2)
        volume = wt.sum() * float(np.prod(spacing))
        f["sphericity"] = float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / max(surface, 1e-6))
        span = np.ptp(np.argwhere(wt), axis=0) + 1
        f["extent"] = float(wt.sum() / np.prod(span))
        f["n_components"] = float(ndimage.label(wt)[1])
    else:
        f.update(elongation=1.0, sphericity=0.0, extent=0.0, n_components=0.0)
    return {k: float(v) for k, v in f.items()}
