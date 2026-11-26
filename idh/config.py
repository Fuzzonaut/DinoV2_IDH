"""Experiment settings. The defaults are the ones used for the reported Kaggle run."""

from dataclasses import dataclass

# Data sources (pinned for reproducibility).
DATASET_REPO = "MedOtter/UCSF-PDGM"  # structural-MRI mirror of UCSF-PDGM v5 on Hugging Face
DATASET_REVISION = "e9372219cf1cd2fdd52260cd45f7514b4aa7638e"
EXPECTED_PATIENTS = 495
METADATA_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata_v5.csv"
METADATA_SHA256 = "afc1c23a0eb0597b56e603741e72f2308b4860281187cd3808e05b7a0116fd78"

CLASS_NAMES = ["IDH wildtype", "IDH mutant"]


@dataclass(frozen=True)
class ExtractionConfig:
    """Image -> slice-feature settings. Changing any of these requires re-extracting features."""

    seed: int = 2026
    model_id: str = "facebook/dinov2-small"
    model_revision: str = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    modalities: tuple = ("FLAIR", "T1c", "T2")  # stacked as the 3 input channels
    image_size: int = 224
    crop_margin_voxels: int = 5
    slice_batch_size: int = 16
    use_mixed_precision: bool = True
    test_fraction: float = 0.15
    validation_fraction: float = 0.15

    def __post_init__(self):
        assert len(self.modalities) == 3
        assert self.image_size >= 28 and self.image_size % 14 == 0
        assert 0 < self.test_fraction + self.validation_fraction < 1


@dataclass(frozen=True)
class SearchConfig:
    """Feature -> patient settings: slice-count search, head architecture and fitting."""

    seed: int = 2026
    counts: tuple = (4, 8, 12, 16, 24, 32, 64, None)  # None = all eligible slices
    refine: bool = True
    cv_folds: int = 3
    cv_repeats: int = 1  # 1 reproduces the reported run; >1 averages over reshuffled folds
    inner_stop_fraction: float = 0.20
    near_tie_auc: float = 0.005
    feature_kind: str = "cls"  # cls | patch_mean | concat
    lower_fraction: float = 0.05
    upper_fraction: float = 0.95
    attention_dim: int = 128
    dropout: float = 0.30
    batch_size: int = 16
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    bootstrap_repeats: int = 1000

    def __post_init__(self):
        assert self.cv_folds >= 2 and self.cv_repeats >= 1
        assert self.feature_kind in {"cls", "patch_mean", "concat"}
        assert 0 <= self.lower_fraction < self.upper_fraction <= 1
        assert len(set(self.counts)) == len(self.counts)


# --------------------------------------------------------------------------- #
# v2: tumour segmentation + tumour-focused classifiers
# --------------------------------------------------------------------------- #
SEG_MODALITIES = ("FLAIR", "T1", "T1c", "T2")  # segmentation input channels


@dataclass(frozen=True)
class SegConfig:
    """3-D tumour segmentation (SegResNet), cross-fitted so no patient is segmented by a model trained on it."""

    seed: int = 2026
    folds: int = 3
    epochs: int = 50
    roi: tuple = (128, 128, 128)
    batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    init_filters: int = 16
    tumour_crop_probability: float = 0.6  # share of training crops centred on a tumour voxel
    sw_overlap: float = 0.5


@dataclass(frozen=True)
class TumourConfig:
    """How the tumour mask is used to build the DINOv2 input."""

    min_tumour_voxels: int = 20  # a slice counts as "tumour slice" above this many WT voxels
    margin_voxels: int = 12  # around the in-plane tumour bounding box
    min_crop_voxels: int = 64  # smallest crop side (1 mm voxels), so tiny tumours keep some context
    image_size: int = 224


@dataclass(frozen=True)
class CVConfig:
    """Repeated stratified CV on the development patients (train + validation); test stays untouched."""

    seed: int = 2026
    folds: int = 5
    repeats: int = 3
    # attention head
    counts: tuple = (8, 16, 32, None)  # tumour slices per patient (None = all)
    max_epochs: int = 40
    attention_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    bag_sampling: bool = True  # train on a random subset of each patient's slices every epoch
    bootstrap_repeats: int = 2000
