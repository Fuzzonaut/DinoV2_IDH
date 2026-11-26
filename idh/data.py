"""Cohort construction: labels from the official metadata, images from the MRI mirror, patient split.

Kept as in the original notebook:
* labels: IDH wildtype = 0, IDH mutant (any recognised IDH1/IDH2 variant) = 1, anything else excluded;
* follow-up examinations are dropped so every patient appears once;
* every patient needs all three contrasts (FLAIR, T1c, T2);
* stratified patient split 70 / 15 / 15 (train / validation / test) with seeds `seed` and `seed + 1`.
"""

import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import DATASET_REPO, DATASET_REVISION, EXPECTED_PATIENTS, METADATA_SHA256, METADATA_URL
from .utils import sha256_file

# Old follow-up exam numbers -> baseline patient number (from the UCSF-PDGM documentation).
LEGACY_FOLLOWUPS = {315: 433, 278: 431, 175: 396, 138: 429, 181: 409, 289: 391}
ID_PATTERN = re.compile(r"UCSF[-_]PDGM[-_](\d{3,4})(_FU\d+d)?", re.I)

WILDTYPE = {"wildtype", "wild-type", "wild type", "idh-wildtype"}
MUTANT = {
    "mutated (nos)", "mutant", "mutated", "idh-mutant",
    "idh1 p.r132h", "idh1 p.r132c", "idh1 p.r132g", "idh1 p.r132s",
    "idh1 p.arg132his", "idh2 p.r172k", "idh2 p.arg172trp",
}

# Accepted file-name suffixes per modality, in order of preference.
MODALITY_SUFFIXES = {
    "FLAIR": ("FLAIR_bias", "FLAIR"),
    "T1c": ("T1c_bias", "T1gad_bias", "T1c", "T1gad"),
    "T2": ("T2_bias", "T2"),
}


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def parse_exam_id(value):
    """'UCSF-PDGM-0429_FU003d' -> ('UCSF-PDGM-0429', True)."""
    match = ID_PATTERN.search(str(value))
    if match is None:
        raise ValueError(f"Unrecognised UCSF-PDGM identifier: {value!r}")
    number = int(match.group(1))
    followup = bool(match.group(2)) or number in LEGACY_FOLLOWUPS
    return f"UCSF-PDGM-{LEGACY_FOLLOWUPS.get(number, number):04d}", followup


def encode_idh(value):
    if pd.isna(value):
        return np.nan
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    if value in WILDTYPE:
        return 0
    if value in MUTANT:
        return 1
    return np.nan


def download_metadata(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the official UCSF-PDGM metadata ...")
        response = requests.get(METADATA_URL, timeout=(15, 90))
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("The metadata URL did not return a CSV file.")
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(response.content)
        tmp.replace(path)
    if sha256_file(path) != METADATA_SHA256:
        warnings.warn("The metadata CSV differs from the version used for the reported results.")
    return path


def load_metadata(path):
    """Return (labelled baseline patients, excluded rows)."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    parsed = raw["ID"].map(parse_exam_id)
    raw["patient_id"] = parsed.str[0]
    raw["is_followup"] = parsed.str[1]
    raw["label"] = raw["IDH"].map(encode_idh)

    reason = np.where(raw["is_followup"], "repeat examination",
                      np.where(raw["label"].isna(), "unknown IDH label", ""))
    excluded = raw.loc[reason != "", ["ID", "IDH", "patient_id"]].assign(reason=reason[reason != ""])
    cohort = raw.loc[reason == ""].copy()
    if cohort["patient_id"].duplicated().any():
        raise ValueError("Duplicate baseline patients in the metadata.")
    cohort["label"] = cohort["label"].astype(int)
    cohort = cohort.rename(columns={"WHO CNS Grade": "grade", "Age at MRI": "age", "Sex": "sex"})
    return cohort.sort_values("patient_id").reset_index(drop=True), excluded


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def download_mri(image_dir, modalities=("FLAIR", "T1c", "T2"), token=None):
    """Download only the three structural volumes per patient (~4.6 GB instead of the full 142 GB)."""
    from huggingface_hub import snapshot_download

    for attempt in range(3):
        try:
            snapshot_download(
                repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
                local_dir=str(image_dir), allow_patterns=[f"*_{m}.nii.gz" for m in modalities],
                max_workers=1, token=token,
            )
            break
        except Exception:
            if attempt == 2:
                raise
            print("Download interrupted, retrying (finished files are kept) ...")
            time.sleep(10)


def index_images(image_dir, cohort, modalities=("FLAIR", "T1c", "T2")):
    """Attach one NIfTI path per modality to each patient. Returns (complete patients, missing)."""
    candidates = {}
    for path in sorted(Path(image_dir).rglob("*.nii*")):
        if not path.name.lower().endswith((".nii", ".nii.gz")) or ID_PATTERN.search(path.name) is None:
            continue
        patient, followup = parse_exam_id(path.name)
        if followup:
            continue
        stem = re.sub(r"\.nii(\.gz)?$", "", path.name, flags=re.I).lower()
        for modality in modalities:
            for rank, suffix in enumerate(MODALITY_SUFFIXES[modality]):
                if stem.endswith("_" + suffix.lower()):
                    candidates.setdefault((patient, modality), []).append((rank, path.resolve()))
                    break

    rows, missing = [], []
    for source in cohort.to_dict("records"):
        row = {k: source.get(k, np.nan) for k in ("patient_id", "label", "grade", "age", "sex")}
        absent = []
        for modality in modalities:
            options = candidates.get((source["patient_id"], modality))
            if not options:
                absent.append(modality)
                continue
            best_rank = min(rank for rank, _ in options)
            best = sorted({p for rank, p in options if rank == best_rank})
            if len(best) != 1:
                raise ValueError(f"Ambiguous {modality} file for {source['patient_id']}: {best}")
            row[modality] = str(best[0])
        if absent:
            missing.append({"patient_id": source["patient_id"], "missing": ", ".join(absent)})
        else:
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(missing, columns=["patient_id", "missing"])


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_patients(cohort, test_fraction=0.15, validation_fraction=0.15, seed=2026):
    """Stratified patient-level split. Adds a `split` column (train / validation / test)."""
    cohort = cohort.sort_values("patient_id").reset_index(drop=True)
    development, test = train_test_split(
        cohort, test_size=test_fraction, stratify=cohort["label"], random_state=seed)
    train, validation = train_test_split(
        development, test_size=validation_fraction / (1 - test_fraction),
        stratify=development["label"], random_state=seed + 1)
    split = pd.Series("train", index=cohort.index)
    split[validation.index] = "validation"
    split[test.index] = "test"
    return cohort.assign(split=split)
