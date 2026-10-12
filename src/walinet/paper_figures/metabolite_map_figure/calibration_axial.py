"""Axial multi-subject comparison for the WALINET-calibration experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.processing import resample_from_to
import numpy as np

from .figure import (
    _map_dir,
    _positive_limit,
    _resolve_array_orientation,
    _transform_spatial,
)


def _load_subject(
    *,
    b0_root: Path,
    no_b0_root: Path,
    subject: str,
    pipelines: Sequence[tuple[str, str]],
    crop_padding: int,
):
    b0_dir = b0_root / subject
    no_b0_dir = no_b0_root / subject
    source_maps = no_b0_dir / "OriginalData/maps"
    t1_path = source_maps / "magnitude.nii.gz"
    mask_path = source_maps / "mask.nii.gz"
    array_mask_path = b0_dir / "masks/brain_mask.npy"
    for path in (t1_path, mask_path, array_mask_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    t1_image = nib.load(t1_path)
    t1 = np.asarray(t1_image.dataobj, dtype=np.float32)
    mask_image = nib.load(mask_path)
    mask = np.asarray(mask_image.dataobj).squeeze() > 0
    array_mask = np.asarray(np.load(array_mask_path)).squeeze() > 0
    orientation_dice, orientation_transform = _resolve_array_orientation(
        array_mask, mask
    )

    native_mask = np.asarray(
        resample_from_to(mask_image, t1_image, order=0).dataobj
    ).squeeze() > 0
    coordinates = np.argwhere(native_mask)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    bounds = tuple(
        (
            max(0, int(lo) - crop_padding),
            min(t1.shape[axis], int(hi) + crop_padding + 1),
        )
        for axis, (lo, hi) in enumerate(zip(low, high))
    )

    mask_coordinates = np.argwhere(mask)
    mrsi_center = np.rint(
        (mask_coordinates.min(axis=0) + mask_coordinates.max(axis=0)) / 2
    ).astype(int)
    world_center = apply_affine(mask_image.affine, mrsi_center)
    native_center = np.rint(
        apply_affine(np.linalg.inv(t1_image.affine), world_center)
    ).astype(int)
    native_center = np.clip(native_center, 0, np.asarray(t1.shape) - 1)

    (x0, x1), (y0, y1), _ = bounds
    axial_t1 = t1[x0:x1, y0:y1, native_center[2]].T
    mrsi_to_native = np.linalg.inv(t1_image.affine) @ mask_image.affine
    native_low = apply_affine(mrsi_to_native, np.full(3, -0.5))
    native_high = apply_affine(mrsi_to_native, np.asarray(mask.shape) - 0.5)
    axial_extent = (
        native_low[0] - x0,
        native_high[0] - x0,
        native_low[1] - y0,
        native_high[1] - y0,
    )
    t1_values = t1[np.isfinite(t1) & (t1 > 0)]
    return {
        "name": subject,
        "t1_path": t1_path,
        "t1_shape": t1.shape,
        "t1_zooms": tuple(float(v) for v in t1_image.header.get_zooms()[:3]),
        "t1_vmax": float(np.percentile(t1_values, 99.5)),
        "axial_t1": axial_t1,
        "axial_extent": axial_extent,
        "mask": mask,
        "mrsi_center": mrsi_center,
        "orientation_dice": orientation_dice,
        "orientation_transform": orientation_transform,
        "map_dirs": [(label, _map_dir(b0_dir / folder)) for label, folder in pipelines],
    }


def create_axial_subject_comparison(
    *,
    b0_root: str | Path,
    no_b0_root: str | Path,
    subjects: Sequence[str],
    subject_display_labels: Mapping[str, str] | None = None,
    pipelines: Sequence[tuple[str, str]],
    metabolites: Mapping[str, str],
    vmax_percentile: float | None = 95,
    crop_padding_t1_voxels: int = 10,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (18.5, 9.4),
):
    """Create native-T1 axial panels with opaque, original-grid LCModel maps."""
    b0_root = Path(b0_root).expanduser().resolve()
    no_b0_root = Path(no_b0_root).expanduser().resolve()
    if not subjects or not pipelines or not metabolites:
        raise ValueError("subjects, pipelines, and metabolites must be non-empty")
    if crop_padding_t1_voxels < 0:
        raise ValueError("crop_padding_t1_voxels must be non-negative")

    loaded = [
        _load_subject(
            b0_root=b0_root,
            no_b0_root=no_b0_root,
            subject=subject,
            pipelines=pipelines,
            crop_padding=crop_padding_t1_voxels,
        )
        for subject in subjects
    ]
    n_methods = len(pipelines)
    figure = plt.figure(figsize=figsize)

    # Explicit panel geometry: image dimensions are fixed, while horizontal
    # whitespace is controlled independently and cannot be reintroduced by a
    # layout engine enforcing equal GridSpec cells.
    panel_height = 0.155
    # Match every axes box to the physical aspect ratio of its displayed T1.
    # Otherwise imshow(aspect="equal") shrinks the visible image inside a wider
    # axes slot and silently reintroduces large white gaps between panels.
    panel_widths = [
        panel_height
        * figsize[1] / figsize[0]
        * state["axial_t1"].shape[1] / state["axial_t1"].shape[0]
        for state in loaded
    ]
    # The native-T1 and MRSI extents leave an invisible lateral margin inside
    # each axes. Slightly overlapping those axes removes that margin from the
    # *visible* layout without changing either image size or crop.
    within_subject_gaps = [-0.009, -0.014, -0.014]
    between_subject_gap = 0.002
    row_gap = -0.018
    first_x = 0.032
    first_row_y = 0.590
    x_positions = []
    cursor = first_x
    for subject_index in range(len(subjects)):
        panel_width = panel_widths[subject_index]
        within_subject_gap = within_subject_gaps[
            min(subject_index, len(within_subject_gaps) - 1)
        ]
        subject_positions = []
        for method_index in range(n_methods):
            subject_positions.append(cursor)
            cursor += panel_width
            if method_index < n_methods - 1:
                cursor += within_subject_gap
        x_positions.append(subject_positions)
        if subject_index < len(subjects) - 1:
            cursor += between_subject_gap

    for subject_index, state in enumerate(loaded):
        panel_width = panel_widths[subject_index]
        group_left = x_positions[subject_index][0]
        group_right = x_positions[subject_index][-1] + panel_width
        group_center = 0.5 * (group_left + group_right)
        axis = figure.add_axes(
            [group_center - panel_width / 2, 0.810, panel_width, 0.165]
        )
        axis.imshow(
            state["axial_t1"],
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=state["t1_vmax"],
            interpolation="nearest",
        )
        display_label = (subject_display_labels or {}).get(
            state["name"], state["name"].replace("/", " ")
        )
        axis.set_title(display_label, fontsize=11, fontweight="bold")
        axis.axis("off")

    for subject_index in range(len(subjects)):
        panel_width = panel_widths[subject_index]
        for method_index, (method_label, _) in enumerate(pipelines):
            if method_label.startswith("U-Net ("):
                shown_label = method_label.replace(" (", "\n(").replace(
                    " training)", "\ntraining)"
                )
            else:
                shown_label = method_label.replace(" ", "\n", 1)
            figure.text(
                x_positions[subject_index][method_index] + panel_width / 2,
                0.757,
                shown_label,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                linespacing=0.95,
            )

    for row_offset, (display_name, lcmodel_name) in enumerate(metabolites.items()):
        row_y = first_row_y - row_offset * (panel_height + row_gap)
        maps = []
        for state in loaded:
            subject_maps = []
            for _, map_dir in state["map_dirs"]:
                map_path = map_dir / f"{lcmodel_name}_amp_map.nii.gz"
                if not map_path.is_file():
                    raise FileNotFoundError(map_path)
                source = np.asarray(
                    nib.load(map_path).dataobj, dtype=np.float32
                ).squeeze()
                source = _transform_spatial(
                    source, **state["orientation_transform"]
                )
                if source.shape != state["mask"].shape:
                    raise ValueError(
                        f"{map_path}: oriented shape {source.shape} != "
                        f"mask {state['mask'].shape}"
                    )
                subject_maps.append(source)
            maps.append(subject_maps)
        vmax = _positive_limit(
            [volume for subject_maps in maps for volume in subject_maps],
            vmax_percentile,
        )

        figure.text(
            0.014, row_y + panel_height / 2, display_name,
            ha="center", va="center", rotation=90,
            fontsize=12, fontweight="bold",
        )

        image = None
        for subject_index, (state, subject_maps) in enumerate(zip(loaded, maps)):
            panel_width = panel_widths[subject_index]
            z = int(state["mrsi_center"][2])
            mask_view = state["mask"][:, :, z].T
            for method_index, source in enumerate(subject_maps):
                axis = figure.add_axes([
                    x_positions[subject_index][method_index],
                    row_y,
                    panel_width,
                    panel_height,
                ])
                axis.imshow(
                    state["axial_t1"],
                    origin="lower",
                    cmap="gray",
                    vmin=0,
                    vmax=state["t1_vmax"],
                    interpolation="nearest",
                )
                source_view = source[:, :, z].T
                shown = np.ma.masked_where(
                    ~mask_view | ~np.isfinite(source_view), source_view
                )
                image = axis.imshow(
                    shown,
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    alpha=1.0,
                    interpolation="nearest",
                    extent=state["axial_extent"],
                )
                axis.axis("off")

        colorbar_height = 0.112
        colorbar_y = row_y + (panel_height - colorbar_height) / 2
        colorbar_axis = figure.add_axes([
            cursor - 0.001, colorbar_y, 0.007, colorbar_height
        ])
        colorbar = figure.colorbar(
            image, cax=colorbar_axis, label="Amplitude [a.u.]"
        )
        colorbar.locator = MaxNLocator(nbins=2)
        colorbar.update_ticks()

    qc = {
        state["name"]: {
            "native_t1_path": state["t1_path"],
            "native_t1_shape": state["t1_shape"],
            "native_t1_zooms": state["t1_zooms"],
            "mrsi_center": tuple(int(v) for v in state["mrsi_center"]),
            "orientation_dice": float(state["orientation_dice"]),
            "map_directories": dict(state["map_dirs"]),
        }
        for state in loaded
    }
    return figure, qc
