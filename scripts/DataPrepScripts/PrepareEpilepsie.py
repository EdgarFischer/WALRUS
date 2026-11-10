#!/usr/bin/env python3

from pathlib import Path
from itertools import permutations, product

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.io import loadmat


BASE_DIR = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/WALRUS/data/7T/NoB0Correction/Epilepsie"
)

SUBJECTS = [
    "EPI_0001",
    "EPI_0002"
]

MINIMUM_ORIENTATION_DICE = 0.99


def _to_complex_array(raw):
    raw = np.asarray(raw)

    if raw.dtype.names is not None and {"real", "imag"}.issubset(raw.dtype.names):
        return raw["real"] + 1j * raw["imag"]

    return raw


def _find_file(folder, *names):
    for name in names:
        path = Path(folder) / name
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Keine dieser Dateien gefunden in {folder}: {', '.join(names)}"
    )


def load_combined_csi(mat_path):
    try:
        with h5py.File(mat_path, "r") as f:
            raw = f["csi"]["Data"][:]
            data = _to_complex_array(raw)
            mask = f["mask"][:]

        print("  Loaded CombinedCSI.mat via h5py")

    except OSError:
        print("  h5py failed; loading via scipy.io.loadmat")

        mat = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
        )

        csi = mat["csi"]

        if hasattr(csi, "Data"):
            raw = csi.Data
        else:
            raw = csi["Data"]

        data = _to_complex_array(raw)
        mask = mat["mask"]

    return np.asarray(data), np.asarray(mask)


def dice_score(first, second):
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    denominator = np.count_nonzero(first) + np.count_nonzero(second)
    if denominator == 0:
        return 1.0
    return float(2 * np.count_nonzero(first & second) / denominator)


def apply_orientation(array, permutation, flips):
    """Apply one lossless spatial-axis permutation and optional flips."""
    result = np.transpose(np.asarray(array), axes=permutation)
    for axis, flip in enumerate(flips):
        if flip:
            result = np.flip(result, axis=axis)
    return result


def find_nifti_to_csi_orientation(nifti_brain_mask, csi_brain_mask):
    """Determine NIfTI-to-CSI orientation from the two brain masks."""
    candidates = []
    for permutation in permutations(range(3)):
        output_shape = tuple(
            nifti_brain_mask.shape[index]
            for index in permutation
        )
        if output_shape != csi_brain_mask.shape:
            continue

        for flips in product((False, True), repeat=3):
            transformed = apply_orientation(
                nifti_brain_mask,
                permutation,
                flips,
            )
            candidates.append(
                (
                    dice_score(transformed, csi_brain_mask),
                    permutation,
                    flips,
                )
            )

    if not candidates:
        raise ValueError(
            "No spatial-axis permutation maps the NIfTI mask shape "
            f"{nifti_brain_mask.shape} to CSI shape {csi_brain_mask.shape}."
        )

    best = max(candidates, key=lambda item: item[0])
    if best[0] < MINIMUM_ORIENTATION_DICE:
        raise ValueError(
            "Could not orient maps/mask consistently with the brain mask "
            f"from CombinedCSI.mat (best Dice={best[0]:.6f}, required "
            f">={MINIMUM_ORIENTATION_DICE:.2f})."
        )
    return best


def load_orientation_reference(maps_dir, csi_brain_mask):
    reference_mask_path = _find_file(
        maps_dir,
        "mask.nii.gz",
        "mask.nii",
    )
    reference_mask_nii = nib.load(reference_mask_path)
    nifti_brain_mask = (
        np.asarray(reference_mask_nii.get_fdata()) > 0.5
    )
    score, permutation, flips = find_nifti_to_csi_orientation(
        nifti_brain_mask,
        csi_brain_mask,
    )
    return reference_mask_nii, score, permutation, flips


def load_magnitude_downsampled(
    maps_dir,
    reference_mask_nii,
    permutation,
    flips,
):
    magnitude_path = _find_file(
        maps_dir,
        "magnitude.nii.gz",
        "magnitude.nii",
    )

    magnitude_nii = nib.load(magnitude_path)

    # Resample the anatomical magnitude into the NIfTI MRSI-mask grid.
    magnitude_resampled = resample_from_to(
        magnitude_nii,
        reference_mask_nii,
        order=1,
    )

    magnitude = magnitude_resampled.get_fdata(dtype=np.float32)

    # Then apply the exact NIfTI-to-CSI transform established by the
    # independent brain masks.
    magnitude = apply_orientation(
        magnitude,
        permutation,
        flips,
    )

    return magnitude.astype(np.float32)


def load_lipid_mask(
    maps_dir,
    reference_mask_nii,
    permutation,
    flips,
):
    lipid_path = _find_file(
        maps_dir,
        "mask_lipid.mnc",
        "mask_lipid.nii.gz",
        "mask_lipid.nii",
    )

    lipid_nii = nib.load(lipid_path)

    # The affine, not merely the array shape, defines correspondence with
    # maps/mask. Resample with nearest-neighbour interpolation if necessary.
    if (
        lipid_nii.shape[:3] != reference_mask_nii.shape[:3]
        or not np.allclose(lipid_nii.affine, reference_mask_nii.affine)
    ):
        lipid_nii = resample_from_to(
            lipid_nii,
            reference_mask_nii,
            order=0,
        )

    lipid_mask = np.asarray(lipid_nii.get_fdata()) > 0.5
    lipid_mask = apply_orientation(
        lipid_mask,
        permutation,
        flips,
    )

    return lipid_mask.astype(bool)


def save_qc_plot(
    subject,
    subject_dir,
    data,
    brain_mask,
    lipid_mask,
    magnitude,
):
    z = min(
        data.shape[2] // 2,
        brain_mask.shape[2] - 1,
        lipid_mask.shape[2] - 1,
        magnitude.shape[2] - 1,
    )

    t = min(4, data.shape[3] - 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.abs(data[:, :, z, t]), cmap="plasma")
    axes[0].set_title(f"|data| z={z}, t={t}")

    axes[1].imshow(brain_mask[:, :, z], cmap="gray")
    axes[1].set_title("Brain Mask")

    axes[2].imshow(lipid_mask[:, :, z], cmap="gray")
    axes[2].set_title("Lipid Mask")

    axes[3].imshow(magnitude[:, :, z], cmap="gray")
    axes[3].set_title("Magnitude")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(subject)
    fig.tight_layout()

    fig.savefig(
        subject_dir / "qc_masks.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def process_subject(subject):
    subject_dir = BASE_DIR / subject
    original_dir = subject_dir / "OriginalData"
    maps_dir = original_dir / "maps"
    masks_dir = subject_dir / "masks"
    mat_path = original_dir / "CombinedCSI.mat"

    print(f"\nProcessing {subject}")

    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing file: {mat_path}")

    if not maps_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {maps_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)

    data, mask = load_combined_csi(mat_path)

    if data.ndim != 4:
        raise ValueError(f"Expected 4D CSI data, got {data.shape}")

    if mask.ndim != 3:
        raise ValueError(f"Expected 3D brain mask, got {mask.shape}")

    # Orientierungen aus dem funktionierenden Code
    data_tr = np.transpose(data, (2, 3, 1, 0))
    brain_mask = np.transpose(mask, (1, 2, 0)) > 0.5

    (
        reference_mask_nii,
        orientation_dice,
        permutation,
        flips,
    ) = load_orientation_reference(
        maps_dir,
        brain_mask,
    )

    lipid_mask = load_lipid_mask(
        maps_dir,
        reference_mask_nii,
        permutation,
        flips,
    )
    magnitude = load_magnitude_downsampled(
        maps_dir,
        reference_mask_nii,
        permutation,
        flips,
    )

    print(f"  data:       {data.shape} -> {data_tr.shape}")
    print(f"  brain mask: {mask.shape} -> {brain_mask.shape}")
    print(f"  lipid mask: {lipid_mask.shape}")
    print(f"  magnitude:  {magnitude.shape}")
    print(
        "  orientation: "
        f"permutation={permutation}, flips={flips}, "
        f"brain-mask Dice={orientation_dice:.6f}"
    )

    expected_spatial_shape = data_tr.shape[:-1]
    for name, array in (
        ("brain mask", brain_mask),
        ("lipid mask", lipid_mask),
        ("magnitude", magnitude),
    ):
        if array.shape != expected_spatial_shape:
            raise ValueError(
                f"{name} shape {array.shape} does not match CSI spatial "
                f"shape {expected_spatial_shape}."
            )

    np.save(original_dir / "data.npy", data_tr)
    np.save(original_dir / "magnitude.npy", magnitude)
    np.save(masks_dir / "brain_mask.npy", brain_mask)
    np.save(masks_dir / "lipid_mask.npy", lipid_mask)

    save_qc_plot(
        subject=subject,
        subject_dir=subject_dir,
        data=data_tr,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
        magnitude=magnitude,
    )

    print("  Saved data.npy")
    print("  Saved magnitude.npy")
    print("  Saved masks/brain_mask.npy")
    print("  Saved masks/lipid_mask.npy")
    print("  Saved qc_masks.png")


def main():
    failed = []

    for subject in SUBJECTS:
        try:
            process_subject(subject)
        except Exception as exc:
            failed.append(subject)
            print(f"  ERROR: {exc}")

    if failed:
        print(f"\nFailed subjects: {', '.join(failed)}")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()
