"""MRI volume -> stack of 3-channel axial slices for the encoder.

1. Load each contrast in canonical (RAS) orientation and check that all three share one voxel grid.
2. Normalise each volume separately: nonzero-voxel 1st/99th percentile -> [0, 1].
3. Resample oblique volumes to an axis-aligned grid (all UCSF-PDGM volumes are already axial).
4. Crop to the nonzero brain extent (+ margin); every nonempty axial slice is eligible.
5. Letterbox each slice to 224 x 224 using the physical aspect ratio (no stretching).

"Nonempty" means brain, not tumour: no segmentation mask or label is used here.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to, resample_to_output


def load_canonical_volume(path):
    image = nib.as_closest_canonical(nib.load(str(path)))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI, got {image.shape}: {path}")
    array = image.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite voxel values: {path}")
    return image, array


def normalize_mri(array):
    support = np.abs(array) > 1e-8
    if support.sum() < 100:
        raise ValueError("Too few nonzero voxels.")
    low, high = np.percentile(array[support], [1, 99])
    if high - low < 1e-6:
        raise ValueError("Constant MRI volume.")
    result = np.zeros_like(array, dtype=np.float32)
    result[support] = np.clip((array[support] - low) / (high - low), 0, 1)
    return result, support


def letterbox(tensor, size, pixel_spacing):
    """Resize (N, C, H, W) to fit in size x size keeping the physical aspect ratio, then zero-pad."""
    height, width = tensor.shape[-2:]
    physical_h, physical_w = height * pixel_spacing[0], width * pixel_spacing[1]
    scale = size / max(physical_h, physical_w)
    new_h = max(1, min(size, round(physical_h * scale)))
    new_w = max(1, min(size, round(physical_w * scale)))
    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    dh, dw = size - new_h, size - new_w
    return F.pad(tensor, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))


def prepare_volume(paths, modalities, crop_margin=5):
    """Load, check, normalise and crop one patient. `paths` maps modality -> NIfTI path."""
    arrays, masks, reference = [], [], None
    for modality in modalities:
        image, raw = load_canonical_volume(paths[modality])
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-3):
            raise ValueError(f"{modality} is not on the same voxel grid as {modalities[0]}; register it first.")
        normalized, support = normalize_mri(raw)
        arrays.append(normalized)
        masks.append(support)
    support = np.logical_or.reduce(masks)

    directions = reference.affine[:3, :3] / np.linalg.norm(reference.affine[:3, :3], axis=0)
    oblique = bool(np.max(np.abs(directions - np.diag(np.diag(directions)))) > 1e-4)
    if oblique:
        images = [nib.Nifti1Image(a, reference.affine) for a in arrays]
        target = resample_to_output(images[0], voxel_sizes=reference.header.get_zooms()[:3], order=1)
        grid = (target.shape, target.affine)
        arrays = [target.get_fdata(dtype=np.float32)] + [
            resample_from_to(im, grid, order=1).get_fdata(dtype=np.float32) for im in images[1:]]
        support = resample_from_to(nib.Nifti1Image(support.astype(np.uint8), reference.affine),
                                   grid, order=0).get_fdata() > 0
        reference = target

    occupied = [np.flatnonzero(support.any(axis=tuple(j for j in range(3) if j != axis))) for axis in range(3)]
    if any(len(v) == 0 for v in occupied):
        raise ValueError("Empty volume after preprocessing.")
    starts = [max(0, int(v[0]) - crop_margin) for v in occupied]
    stops = [min(support.shape[i], int(v[-1]) + 1 + crop_margin) for i, v in enumerate(occupied)]
    indices = occupied[2].astype(np.int32)
    if len(indices) < 2:
        raise ValueError("Too few nonempty axial slices.")

    # Physical z position of every eligible slice, and its relative position within the brain (0..1).
    centers = np.column_stack([np.full(len(indices), (starts[0] + stops[0] - 1) / 2),
                               np.full(len(indices), (starts[1] + stops[1] - 1) / 2), indices])
    z_mm = nib.affines.apply_affine(reference.affine, centers)[:, 2]
    fractions = ((z_mm - z_mm.min()) / (z_mm.max() - z_mm.min())).astype(np.float32)
    zooms = [float(v) for v in reference.header.get_zooms()[:3]]
    return {
        "arrays": arrays, "indices": indices, "z_mm": z_mm, "fractions": fractions,
        "area": support.mean(axis=(0, 1))[indices].astype(np.float32),
        "crop_starts": starts, "crop_stops": stops, "voxel_spacing": zooms, "resampled_oblique": oblique,
    }


def image_batch(prepared, positions, image_size=224):
    """Slices at `positions` (indices into prepared['indices']) -> float tensor (N, 3, size, size) in [0, 1]."""
    indices = prepared["indices"][np.asarray(positions, dtype=int)]
    starts, stops = prepared["crop_starts"], prepared["crop_stops"]
    # (x, y, z) arrays -> (N, H=y, W=x) per contrast, stacked as channels.
    stacked = np.stack([
        array[starts[0]:stops[0], starts[1]:stops[1], :][:, :, indices].transpose(2, 1, 0)
        for array in prepared["arrays"]], axis=1)
    zooms = prepared["voxel_spacing"]
    images = letterbox(torch.from_numpy(stacked.copy()), image_size, (zooms[1], zooms[0])).clamp(0, 1)
    if images.shape != (len(indices), 3, image_size, image_size) or not torch.isfinite(images).all():
        raise ValueError("Invalid slice batch after preprocessing.")
    return images
