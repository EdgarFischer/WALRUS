from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


def dilate_lipid_inward(
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
    n_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move mask layers across the brain/lipid boundary.

    Positive values grow the lipid mask into the brain; negative values grow
    the brain mask into the lipid mask. The third return value contains all
    voxels transferred between the masks.
    """
    brain = np.asarray(brain_mask, dtype=bool).copy()
    lipid = np.asarray(lipid_mask, dtype=bool).copy()
    n_voxels = int(n_voxels)
    if brain.shape != lipid.shape:
        raise ValueError("brain_mask and lipid_mask must have the same shape.")
    if np.any(brain & lipid):
        raise ValueError("brain_mask and lipid_mask must initially be disjoint.")

    transferred = np.zeros_like(brain)
    structure = generate_binary_structure(3, 1)
    growing, shrinking = (lipid, brain) if n_voxels > 0 else (brain, lipid)
    for _ in range(abs(n_voxels)):
        new_layer = binary_dilation(growing, structure=structure) & shrinking
        if not np.any(new_layer):
            break
        growing[new_layer] = True
        shrinking[new_layer] = False
        transferred[new_layer] = True

    return brain, lipid, transferred
