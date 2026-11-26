"""Run the trained pipeline on a new patient.

    from idh.inference import IDHPredictor

    predictor = IDHPredictor("model")
    result = predictor.predict({"FLAIR": "flair.nii.gz", "T1c": "t1c.nii.gz", "T2": "t2.nii.gz"})
    print(result["predicted_class"], result["score"])

The three volumes must already be co-registered and skull-stripped (as in UCSF-PDGM).
The score is not a calibrated probability. Research code, not a medical device.
"""

import warnings
from pathlib import Path

import numpy as np
import torch

from .config import CLASS_NAMES, ExtractionConfig
from .features import encode_volume, load_encoder, model_digest, select_feature_kind, select_slices
from .preprocessing import prepare_volume
from .training import load_head
from .utils import read_json


class IDHPredictor:
    def __init__(self, model_dir="model", device=None, hf_token=None):
        model_dir = Path(model_dir)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.info = read_json(model_dir / "config.json")
        saved = {k: v for k, v in self.info["extraction_config"].items() if k in ExtractionConfig.__dataclass_fields__}
        self.config = ExtractionConfig(**{**saved, "modalities": tuple(saved["modalities"])})
        self.head = load_head(model_dir / "head.pt", self.device)
        self._encoder = None
        self._hf_token = hf_token

    @property
    def encoder(self):
        if self._encoder is None:  # downloaded on first use (~90 MB)
            self._encoder = load_encoder(self.config, self.device, token=self._hf_token)
            expected = self.info.get("encoder_sha256")
            if expected and model_digest(self._encoder) != expected:
                warnings.warn("The downloaded DINOv2 weights differ from the ones used in training.")
        return self._encoder

    def predict(self, paths):
        """`paths`: {'FLAIR': ..., 'T1c': ..., 'T2': ...} NIfTI files of one patient."""
        missing = set(self.config.modalities) - set(paths)
        if missing:
            raise ValueError(f"Missing modalities: {sorted(missing)}")
        prepared = prepare_volume(paths, self.config.modalities, self.config.crop_margin_voxels)
        cls, patch = encode_volume(prepared, self.encoder, self.config, self.device)
        vectors = select_feature_kind({"cls_features": cls, "patch_mean_features": patch}, self.info["feature_kind"])
        lower, upper = self.info["slice_range"]
        chosen = select_slices(prepared["fractions"], self.info["selected_count"], lower, upper)

        bag = torch.from_numpy(np.ascontiguousarray(vectors[chosen], dtype=np.float32)).unsqueeze(0).to(self.device)
        mask = torch.ones(bag.shape[:2], dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logit, attention = self.head(bag, mask, return_attention=True)
        score = float(logit.sigmoid())
        threshold = float(self.info["threshold"])
        return {
            "score": score,
            "threshold": threshold,
            "predicted_class": CLASS_NAMES[int(score >= threshold)],
            "n_slices_used": int(len(chosen)),
            "slice_index": prepared["indices"][chosen].tolist(),  # axial index in the canonical (RAS) volume
            "slice_z_mm": prepared["z_mm"][chosen].tolist(),
            "attention": attention[0].cpu().tolist(),
        }
