#!/usr/bin/env python3
"""Orient hand-segmented masks to data.npy and regenerate mask QC plots."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from extract_combined_csi_data_and_masks import (
    apply_orientation,
    find_nifti_to_csi_orientation,
)
from walinet.data.combined_csi_io import load_combined_csi


DATA_ROOT = REPOSITORY_ROOT / "data" / "10.5T"
SUBJECTS = ("Measurement26", "Measurement29", "Measurement31", "Measurement34")


def load_binary_nifti(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask: {path}")
    image = nib.load(path)
    mask = np.squeeze(np.asarray(image.dataobj)) > 0
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3-D mask at {path}, got {mask.shape}.")
    return image, mask


def save_qc(
    subject_dir: Path,
    data_path: Path,
    magnitude_path: Path,
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
    headmask: np.ndarray,
    dice: float,
) -> Path:
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    anatomical_magnitude = np.load(
        magnitude_path, mmap_mode="r", allow_pickle=False
    )
    if data.shape[:3] != brain_mask.shape:
        raise ValueError(
            f"data.npy spatial shape {data.shape[:3]} does not match masks "
            f"{brain_mask.shape}."
        )
    if anatomical_magnitude.shape != brain_mask.shape:
        raise ValueError(
            f"magnitude.npy shape {anatomical_magnitude.shape} does not match masks "
            f"{brain_mask.shape}."
        )

    fid_index = min(3, data.shape[-1] - 1)
    fid_magnitude = np.abs(data[..., fid_index])
    slices = np.flatnonzero(np.any(headmask, axis=(0, 1)))
    if not slices.size:
        raise ValueError(f"Empty hand-segmented headmask for {subject_dir.name}.")

    figure, axes = plt.subplots(
        len(slices),
        5,
        figsize=(15, max(2.2 * len(slices), 3.0)),
        squeeze=False,
    )
    titles = (
        f"|data.npy[..., {fid_index}]|",
        "magnitude.npy",
        "Brain mask (hand-derived)",
        "Lipid mask (hand-segmented)",
        "Headmask = brain OR lipid",
    )

    for row, z_index in enumerate(slices):
        background = np.asarray(fid_magnitude[:, :, z_index])
        finite = background[np.isfinite(background)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        background = np.clip(np.nan_to_num(background), 0, vmax)

        anatomical = np.asarray(anatomical_magnitude[:, :, z_index])
        finite_anatomical = anatomical[np.isfinite(anatomical)]
        anatomical_vmax = (
            float(np.percentile(finite_anatomical, 99))
            if finite_anatomical.size
            else 1.0
        )
        if not np.isfinite(anatomical_vmax) or anatomical_vmax <= 0:
            anatomical_vmax = 1.0
        anatomical = np.clip(
            np.nan_to_num(anatomical), 0, anatomical_vmax
        )

        axes[row, 0].imshow(background, cmap="gray", interpolation="nearest")
        axes[row, 1].imshow(anatomical, cmap="gray", interpolation="nearest")
        axes[row, 2].imshow(
            brain_mask[:, :, z_index], cmap="gray", interpolation="nearest",
            vmin=0, vmax=1,
        )
        axes[row, 3].imshow(
            lipid_mask[:, :, z_index], cmap="gray", interpolation="nearest",
            vmin=0, vmax=1,
        )
        axes[row, 4].imshow(
            headmask[:, :, z_index], cmap="gray", interpolation="nearest",
            vmin=0, vmax=1,
        )

        axes[row, 0].set_ylabel(f"z={z_index}", fontsize=9)
        for column, axis in enumerate(axes[row]):
            if row == 0:
                axis.set_title(titles[column], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f"{subject_dir.name}: final hand-segmented masks in data.npy space "
        f"(orientation-reference Dice={dice:.6f})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output = subject_dir / "qc_data_and_masks.png"
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def process_subject(subject: str) -> None:
    subject_dir = DATA_ROOT / subject
    original_dir = subject_dir / "OriginalData"
    maps_dir = original_dir / "maps"
    masks_dir = subject_dir / "masks"
    data_path = original_dir / "data.npy"
    magnitude_path = original_dir / "magnitude.npy"

    print(f"\nProcessing {subject}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data.npy: {data_path}")

    _, csi_brain = load_combined_csi(original_dir / "CombinedCSI.mat")
    csi_brain = np.squeeze(np.asarray(csi_brain)) > 0.5

    reference_image, reference_brain = load_binary_nifti(maps_dir / "mask.nii.gz")
    brain_image, hand_brain = load_binary_nifti(
        maps_dir / "brain_hand_segmented.nii.gz"
    )
    lipid_image, hand_lipid = load_binary_nifti(
        maps_dir / "lipid_hand_segmented.nii.gz"
    )

    for label, image, mask in (
        ("brain", brain_image, hand_brain),
        ("lipid", lipid_image, hand_lipid),
    ):
        if mask.shape != reference_brain.shape:
            raise ValueError(
                f"{label} hand mask shape {mask.shape} differs from reference "
                f"NIfTI mask shape {reference_brain.shape}."
            )
        if not np.allclose(image.affine, reference_image.affine, rtol=0, atol=1e-5):
            raise ValueError(
                f"{label} hand mask affine differs from maps/mask.nii.gz."
            )

    dice, permutation, flips = find_nifti_to_csi_orientation(
        reference_brain,
        csi_brain,
    )
    brain_mask = apply_orientation(hand_brain, permutation, flips).astype(bool)
    lipid_mask = apply_orientation(hand_lipid, permutation, flips).astype(bool)

    # The lipid ring defines the outer compartment. Small boundary overlaps can
    # arise after manual edits; assign those voxels unambiguously to lipid while
    # leaving the hand-segmented NIfTI source files untouched.
    overlap_voxels = int(np.count_nonzero(brain_mask & lipid_mask))
    if overlap_voxels:
        print(
            f"  Removing {overlap_voxels} lipid-overlap voxel(s) from the "
            "derived brain_mask.npy"
        )
        brain_mask = brain_mask & ~lipid_mask

    headmask = brain_mask | lipid_mask

    magnitude_image = nib.load(maps_dir / "magnitude.nii.gz")
    magnitude_on_mask_grid = resample_from_to(
        magnitude_image,
        reference_image,
        order=1,
    ).get_fdata(dtype=np.float32)
    magnitude = apply_orientation(
        magnitude_on_mask_grid,
        permutation,
        flips,
    ).astype(np.float32, copy=False)

    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if data.shape[:3] != brain_mask.shape:
        raise ValueError(
            f"Oriented mask shape {brain_mask.shape} differs from data.npy "
            f"spatial shape {data.shape[:3]}."
        )
    if magnitude.shape != brain_mask.shape:
        raise ValueError(
            f"Oriented magnitude shape {magnitude.shape} differs from masks "
            f"{brain_mask.shape}."
        )
    masks_dir.mkdir(parents=True, exist_ok=True)
    np.save(masks_dir / "brain_mask.npy", brain_mask)
    np.save(masks_dir / "lipid_mask.npy", lipid_mask)
    np.save(masks_dir / "headmask.npy", headmask)
    np.save(magnitude_path, magnitude)

    qc_path = save_qc(
        subject_dir,
        data_path,
        magnitude_path,
        brain_mask,
        lipid_mask,
        headmask,
        dice,
    )
    metadata = {
        "dataset": subject,
        "mask_source": "hand-segmented NIfTI masks",
        "brain_source": "OriginalData/maps/brain_hand_segmented.nii.gz",
        "lipid_source": "OriginalData/maps/lipid_hand_segmented.nii.gz",
        "data_shape": list(data.shape),
        "mask_shape": list(brain_mask.shape),
        "magnitude_shape": list(magnitude.shape),
        "magnitude_source": "OriginalData/maps/magnitude.nii.gz",
        "nifti_to_csi_permutation": list(permutation),
        "nifti_to_csi_flips": list(flips),
        "orientation_reference_dice": dice,
        "brain_voxels": int(brain_mask.sum()),
        "lipid_voxels": int(lipid_mask.sum()),
        "head_voxels": int(headmask.sum()),
        "removed_brain_lipid_overlap_voxels": overlap_voxels,
    }
    (subject_dir / "qc_data_and_masks.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"  transform: permutation={permutation}, flips={flips}, Dice={dice:.6f}")
    print(f"  brain/lipid/head voxels: {brain_mask.sum()}/{lipid_mask.sum()}/{headmask.sum()}")
    print(f"  saved masks: {masks_dir}")
    print(f"  saved magnitude: {magnitude_path}")
    print(f"  saved QC: {qc_path}")


def main() -> None:
    failures = []
    for subject in SUBJECTS:
        try:
            process_subject(subject)
        except Exception as error:
            failures.append((subject, error))
            print(f"  ERROR: {error}", file=sys.stderr)

    if failures:
        print("\nFailed subjects:", file=sys.stderr)
        for subject, error in failures:
            print(f"  {subject}: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("\nFinished all hand-segmented masks.")


if __name__ == "__main__":
    main()
