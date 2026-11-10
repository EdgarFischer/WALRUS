#!/usr/bin/env python3
"""Extract CSI data and consistently oriented masks from copied datasets."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations, product
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


# Make the script work directly from a WALRUS checkout without installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from walinet.data.combined_csi_io import load_combined_csi


OUTPUT_DATA_NAME = "data.npy"
OUTPUT_BRAIN_NAME = "initial_brain_mask.npy"
OUTPUT_LIPID_NAME = "initial_lipid_mask.npy"
OUTPUT_HEAD_NAME = "headmask.npy"
QC_PLOT_NAME = "qc_data_and_masks.png"
QC_INFO_NAME = "qc_data_and_masks.json"
MINIMUM_DICE = 0.99


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = np.count_nonzero(first) + np.count_nonzero(second)
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(first & second) / denominator)


def apply_orientation(
    array: np.ndarray,
    permutation: tuple[int, int, int],
    flips: tuple[bool, bool, bool],
) -> np.ndarray:
    """Permute spatial axes, then optionally flip each resulting axis."""
    result = np.transpose(np.asarray(array), axes=permutation)
    for axis, flip in enumerate(flips):
        if flip:
            result = np.flip(result, axis=axis)
    return result


def find_nifti_to_csi_orientation(
    nifti_brain_mask: np.ndarray,
    csi_brain_mask: np.ndarray,
) -> tuple[float, tuple[int, int, int], tuple[bool, bool, bool]]:
    """Find the lossless 3-D axis/flip transform from NIfTI to CSI space."""
    candidates = []

    for permutation in permutations(range(3)):
        permuted_shape = tuple(nifti_brain_mask.shape[index] for index in permutation)
        if permuted_shape != csi_brain_mask.shape:
            continue

        for flips in product((False, True), repeat=3):
            transformed = apply_orientation(
                nifti_brain_mask,
                permutation=permutation,
                flips=flips,
            )
            candidates.append(
                (dice_score(transformed, csi_brain_mask), permutation, flips)
            )

    if not candidates:
        raise ValueError(
            "No axis permutation maps the NIfTI mask shape "
            f"{nifti_brain_mask.shape} to the CSI mask shape "
            f"{csi_brain_mask.shape}."
        )

    best = max(candidates, key=lambda item: item[0])
    if best[0] < MINIMUM_DICE:
        raise ValueError(
            "Could not orient maps/mask.nii.gz consistently with the mask "
            f"embedded in CombinedCSI.mat (best Dice={best[0]:.6f}, "
            f"required >= {MINIMUM_DICE:.2f})."
        )

    return best


def load_nifti_masks(
    maps_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    brain_path = maps_dir / "mask.nii.gz"
    lipid_path = maps_dir / "mask_lipid.nii.gz"

    for path in (brain_path, lipid_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required mask: {path}")

    brain_image = nib.load(brain_path)
    lipid_image = nib.load(lipid_path)

    # Lipid and brain masks must first live on the same NIfTI voxel grid.
    if (
        lipid_image.shape[:3] != brain_image.shape[:3]
        or not np.allclose(lipid_image.affine, brain_image.affine)
    ):
        print("  Resampling lipid mask to the NIfTI brain-mask grid")
        lipid_image = resample_from_to(lipid_image, brain_image, order=0)

    nifti_brain = np.squeeze(np.asarray(brain_image.get_fdata()) > 0.5)
    nifti_lipid = np.squeeze(np.asarray(lipid_image.get_fdata()) > 0.5)

    if nifti_brain.ndim != 3 or nifti_lipid.ndim != 3:
        raise ValueError(
            "Expected 3-D NIfTI masks, got "
            f"brain={nifti_brain.shape}, lipid={nifti_lipid.shape}."
        )

    return nifti_brain, nifti_lipid


def add_mask_overlay(axis, background, mask, color, alpha=0.35):
    axis.imshow(background, cmap="gray", interpolation="nearest")
    overlay = np.ma.masked_where(~mask, mask)
    axis.imshow(
        overlay,
        cmap=plt.matplotlib.colors.ListedColormap([color]),
        interpolation="nearest",
        alpha=alpha,
        vmin=0,
        vmax=1,
    )
    if np.any(mask) and not np.all(mask):
        axis.contour(mask, levels=[0.5], colors=[color], linewidths=0.8)


def save_qc_plot(
    subject_dir: Path,
    data_path: Path,
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
    headmask: np.ndarray,
    orientation_dice: float,
) -> Path:
    """Create an all-slice QC plot from the saved data.npy."""
    # Intentionally reload the output: this validates the file used downstream.
    data = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if data.shape[:3] != brain_mask.shape:
        raise ValueError(
            f"Saved data spatial shape {data.shape[:3]} does not match masks "
            f"{brain_mask.shape}."
        )

    fid_index = min(3, data.shape[-1] - 1)
    fid_magnitude = np.abs(data[..., fid_index])
    slice_indices = np.flatnonzero(np.any(headmask, axis=(0, 1)))
    if slice_indices.size == 0:
        raise ValueError("The combined headmask is empty; cannot create QC plot.")

    figure, axes = plt.subplots(
        len(slice_indices),
        4,
        figsize=(12, max(2.2 * len(slice_indices), 3.0)),
        squeeze=False,
    )
    column_titles = (
        f"|data.npy[..., {fid_index}]|",
        "Initial brain mask",
        "Initial lipid mask",
        "Headmask = brain OR lipid",
    )

    for row, z_index in enumerate(slice_indices):
        background = np.asarray(fid_magnitude[:, :, z_index])
        finite = background[np.isfinite(background)]
        upper = float(np.percentile(finite, 99)) if finite.size else 1.0
        if not np.isfinite(upper) or upper <= 0:
            upper = 1.0
        background = np.clip(np.nan_to_num(background), 0, upper)

        axes[row, 0].imshow(background, cmap="gray", interpolation="nearest")
        add_mask_overlay(
            axes[row, 1], background, brain_mask[:, :, z_index], "lime"
        )
        add_mask_overlay(
            axes[row, 2], background, lipid_mask[:, :, z_index], "magenta"
        )
        add_mask_overlay(
            axes[row, 3], background, headmask[:, :, z_index], "cyan"
        )

        axes[row, 0].set_ylabel(f"z={z_index}", fontsize=9)
        for column, axis in enumerate(axes[row]):
            if row == 0:
                axis.set_title(column_titles[column], fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle(
        f"{subject_dir.name}: CSI/NIfTI mask orientation QC "
        f"(brain-mask Dice={orientation_dice:.6f})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_path = subject_dir / QC_PLOT_NAME
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output_path


def process_dataset(subject_dir: Path) -> None:
    original_data_dir = subject_dir / "OriginalData"
    maps_dir = original_data_dir / "maps"
    masks_dir = subject_dir / "masks"
    combined_csi_path = original_data_dir / "CombinedCSI.mat"

    if not combined_csi_path.is_file():
        raise FileNotFoundError(f"Missing CombinedCSI.mat: {combined_csi_path}")
    if not maps_dir.is_dir():
        raise FileNotFoundError(f"Missing maps directory: {maps_dir}")

    print(f"\nProcessing {subject_dir.name}")
    data, embedded_mask = load_combined_csi(combined_csi_path)
    data = np.asarray(data)
    embedded_mask = np.squeeze(np.asarray(embedded_mask)) > 0.5

    if data.ndim != 4:
        raise ValueError(f"Expected 4-D csi.Data, got shape {data.shape}.")
    if not np.iscomplexobj(data):
        raise TypeError(f"Expected complex csi.Data, got dtype {data.dtype}.")
    if embedded_mask.ndim != 3:
        raise ValueError(
            f"Expected a 3-D embedded mask, got shape {embedded_mask.shape}."
        )
    if data.shape[:3] != embedded_mask.shape:
        raise ValueError(
            f"csi.Data spatial shape {data.shape[:3]} does not match the "
            f"embedded mask {embedded_mask.shape}."
        )

    # load_combined_csi reverses raw MATLAB-v7.3/HDF5 axes, exactly matching
    # data.transpose() in InvestigateSpectra. Classic MAT files are already in
    # logical MATLAB order and therefore need no additional transpose.
    data_path = original_data_dir / OUTPUT_DATA_NAME
    np.save(data_path, data)
    del data

    nifti_brain, nifti_lipid = load_nifti_masks(maps_dir)
    score, permutation, flips = find_nifti_to_csi_orientation(
        nifti_brain,
        embedded_mask,
    )
    oriented_nifti_brain = apply_orientation(nifti_brain, permutation, flips)
    initial_lipid_mask = apply_orientation(nifti_lipid, permutation, flips)

    # Use the embedded mask as the final brain mask. The NIfTI version is the
    # independent reference used to determine and verify lipid-mask orientation.
    initial_brain_mask = embedded_mask.astype(bool, copy=False)
    initial_lipid_mask = initial_lipid_mask.astype(bool, copy=False)
    headmask = initial_brain_mask | initial_lipid_mask

    if not np.array_equal(oriented_nifti_brain, initial_brain_mask):
        disagreement = int(
            np.count_nonzero(oriented_nifti_brain != initial_brain_mask)
        )
        print(
            f"  WARNING: Dice passed, but brain masks differ in "
            f"{disagreement} voxel(s)."
        )

    masks_dir.mkdir(parents=True, exist_ok=True)
    np.save(masks_dir / OUTPUT_BRAIN_NAME, initial_brain_mask)
    np.save(masks_dir / OUTPUT_LIPID_NAME, initial_lipid_mask)
    np.save(masks_dir / OUTPUT_HEAD_NAME, headmask)

    qc_plot_path = save_qc_plot(
        subject_dir=subject_dir,
        data_path=data_path,
        brain_mask=initial_brain_mask,
        lipid_mask=initial_lipid_mask,
        headmask=headmask,
        orientation_dice=score,
    )

    qc_info = {
        "dataset": subject_dir.name,
        "data_shape": list(np.load(data_path, mmap_mode="r").shape),
        "mask_shape": list(initial_brain_mask.shape),
        "nifti_to_csi_permutation": list(permutation),
        "nifti_to_csi_flips": list(flips),
        "brain_mask_dice": score,
        "brain_voxels": int(np.count_nonzero(initial_brain_mask)),
        "lipid_voxels": int(np.count_nonzero(initial_lipid_mask)),
        "head_voxels": int(np.count_nonzero(headmask)),
    }
    qc_info_path = subject_dir / QC_INFO_NAME
    qc_info_path.write_text(json.dumps(qc_info, indent=2) + "\n")

    print(f"  data.npy:              {data_path}")
    print(f"  data shape/dtype:       {qc_info['data_shape']} / complex")
    print(f"  NIfTI -> CSI transform: permutation={permutation}, flips={flips}")
    print(f"  brain-mask Dice:        {score:.6f}")
    print(f"  masks:                  {masks_dir}")
    print(f"  QC plot:                {qc_plot_path}")
    print(f"  QC metadata:            {qc_info_path}")


def discover_datasets(root: Path, names: list[str] | None) -> list[Path]:
    root = root.expanduser().resolve()
    if (root / "OriginalData").is_dir():
        if names:
            raise ValueError(
                "Do not use --datasets when TARGET_ROOT is already one dataset."
            )
        return [root]

    if names:
        datasets = [root / name for name in names]
    else:
        datasets = sorted(
            path for path in root.iterdir() if (path / "OriginalData").is_dir()
        )

    if not datasets:
        raise FileNotFoundError(
            f"No dataset directories containing OriginalData found below {root}."
        )
    return datasets


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract data.npy and consistently oriented initial brain/lipid/"
            "head masks from copied CombinedCSI datasets."
        )
    )
    parser.add_argument(
        "target_root",
        type=Path,
        help="Parent containing dataset folders, or one dataset folder itself.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset folder names below TARGET_ROOT.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    datasets = discover_datasets(arguments.target_root, arguments.datasets)
    print(f"Discovered {len(datasets)} dataset(s).")

    failed = []
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as error:
            failed.append((dataset, error))
            print(f"  ERROR: {error}", file=sys.stderr)

    if failed:
        print("\nFailed datasets:", file=sys.stderr)
        for dataset, error in failed:
            print(f"  {dataset}: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nFinished {len(datasets)} dataset(s).")


if __name__ == "__main__":
    main()
