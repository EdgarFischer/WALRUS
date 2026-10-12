#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from scipy.io import loadmat
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib.pyplot as plt


BASE_DIR = Path("/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/WALRUS/data/7T/NoB0Correction/Tumor")

SUBJECTS = [
    "Tumor_Patient_160_B_2026-03-17",
    #"MS_230",
    #"MS_250",
    # "MS_260",
    # "MS_280",
    # "MS_320",
    # "MS_340",
    # "MS_400",
    # "MS_430",
]


def process_subject(subject: str):
    subject_dir = BASE_DIR / subject
    original_dir = subject_dir / "OriginalData"
    maps_dir = original_dir / "maps"
    masks_dir = subject_dir / "masks"

    mat_path = original_dir / "CombinedCSI.mat"

    print(f"\nProcessing {subject}")

    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing file: {mat_path}")

    if not maps_dir.is_dir():
        raise FileNotFoundError(f"Missing maps directory: {maps_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)

    mat = loadmat(mat_path)

    data = mat["csi"]["Data"][0, 0]
    brain_mask = mat["mask"]

    mag_nii = nib.load(maps_dir / "magnitude.nii")
    mask_nii = nib.load(maps_dir / "mask.nii")

    mag_resampled_nii = resample_from_to(mag_nii, mask_nii, order=1)
    magnitude_down = mag_resampled_nii.get_fdata(dtype=np.float32)
    magnitude_down = magnitude_down[::-1, ::-1, :]

    lipid_nii = nib.load(maps_dir / "mask_lipid.nii")
    lipid_mask = lipid_nii.get_fdata()
    lipid_mask = lipid_mask[::-1, ::-1, :]

    np.save(original_dir / "data.npy", data)
    np.save(original_dir / "magnitude.npy", magnitude_down)
    np.save(masks_dir / "brain_mask.npy", brain_mask)
    np.save(masks_dir / "lipid_mask.npy", lipid_mask)

    save_qc_plot(
        subject=subject,
        subject_dir=subject_dir,
        data=data,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
        magnitude_down=magnitude_down,
    )

    print(f"Saved data.npy")
    print(f"Saved magnitude.npy")
    print(f"Saved masks/brain_mask.npy")
    print(f"Saved masks/lipid_mask.npy")
    print(f"Saved QC plot")


def save_qc_plot(subject, subject_dir, data, brain_mask, lipid_mask, magnitude_down):
    z = data.shape[2] // 2
    t = min(4, data.shape[3] - 1)

    data_slice = np.abs(data[:, :, z, t])
    brain_slice = brain_mask[:, :, z]
    lipid_slice = lipid_mask[:, :, z]
    mag_slice = magnitude_down[:, :, z]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(data_slice, cmap="plasma")
    axes[0].set_title(f"|data| z={z}, t={t}")
    axes[0].axis("off")

    axes[1].imshow(brain_slice, cmap="gray")
    axes[1].set_title(f"brain_mask z={z}")
    axes[1].axis("off")

    axes[2].imshow(lipid_slice, cmap="gray")
    axes[2].set_title(f"lipid_mask z={z}")
    axes[2].axis("off")

    axes[3].imshow(mag_slice, cmap="gray")
    axes[3].set_title(f"magnitude_down z={z}")
    axes[3].axis("off")

    fig.suptitle(subject)
    fig.tight_layout()

    out_path = subject_dir / "qc_masks.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for subject in SUBJECTS:
        process_subject(subject)

    print("\nDone.")


if __name__ == "__main__":
    main()