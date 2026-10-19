"""Create an anatomical localizer plus two-slice metabolite-map comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np


def _map_dir(path: Path) -> Path:
    for candidate in (path, path / "maps/Orig", path / "Orig"):
        if candidate.is_dir() and any(candidate.glob("*_amp_map.nii.gz")):
            return candidate
    raise FileNotFoundError(f"No LCModel amplitude maps below {path}")


def _fine_reference(reference: nib.spatialimages.SpatialImage, factor: int):
    if factor < 1:
        raise ValueError("anatomy_upsampling must be >= 1")
    shape = np.asarray(reference.shape[:3], dtype=int)
    affine = np.asarray(reference.affine, dtype=float).copy()
    fine_affine = affine.copy()
    fine_affine[:3, :3] = affine[:3, :3] / factor
    # Preserve the outer voxel boundaries of the original MRSI field of view.
    fine_affine[:3, 3] = (
        affine[:3, 3]
        - affine[:3, :3] @ np.full(3, (factor - 1) / (2 * factor))
    )
    return tuple((shape * factor).tolist()), fine_affine


def _fine_index(index: int, factor: int) -> int:
    return int(round((index + 0.5) * factor - 0.5))


def _positive_limit(volumes: Sequence[np.ndarray], percentile: float | None) -> float:
    values = [volume[np.isfinite(volume) & (volume > 0)] for volume in volumes]
    values = [value for value in values if value.size]
    if not values:
        return 1.0
    joined = np.concatenate(values)
    result = joined.max() if percentile is None else np.percentile(joined, percentile)
    return float(result) if np.isfinite(result) and result > 0 else 1.0


def _transform_spatial(array, *, swap_xy, flip_x, flip_y, flip_z):
    result = np.swapaxes(array, 0, 1) if swap_xy else array
    if flip_x:
        result = np.flip(result, axis=0)
    if flip_y:
        result = np.flip(result, axis=1)
    if flip_z:
        result = np.flip(result, axis=2)
    return result


def _transform_index(index, shape, *, swap_xy, flip_x, flip_y, flip_z):
    """Apply ``_transform_spatial`` to one voxel index instead of an array."""
    coordinates = [int(value) for value in index]
    transformed_shape = list(shape)
    if len(coordinates) != 3:
        raise ValueError(f"Expected a 3-D voxel index, got {index}")
    if swap_xy:
        coordinates[0], coordinates[1] = coordinates[1], coordinates[0]
        transformed_shape[0], transformed_shape[1] = (
            transformed_shape[1], transformed_shape[0]
        )
    for axis, flip in enumerate((flip_x, flip_y, flip_z)):
        if flip:
            coordinates[axis] = transformed_shape[axis] - 1 - coordinates[axis]
    return tuple(coordinates)


def _resolve_array_orientation(array_mask: np.ndarray, nifti_mask: np.ndarray):
    """Find the lossless axis/flip transform mapping NumPy data to NIfTI space."""
    best = None
    for swap_xy in (False, True):
        for flip_x in (False, True):
            for flip_y in (False, True):
                for flip_z in (False, True):
                    settings = dict(swap_xy=swap_xy, flip_x=flip_x,
                                    flip_y=flip_y, flip_z=flip_z)
                    candidate = _transform_spatial(array_mask, **settings)
                    if candidate.shape != nifti_mask.shape:
                        continue
                    denominator = np.count_nonzero(candidate) + np.count_nonzero(nifti_mask)
                    dice = (2 * np.count_nonzero(candidate & nifti_mask) / denominator
                            if denominator else 1.0)
                    if best is None or dice > best[0]:
                        best = (dice, settings)
    if best is None or best[0] < 0.99:
        score = float("nan") if best is None else best[0]
        raise ValueError(
            "Could not align brain_mask.npy with mask.nii.gz using axis/flip "
            f"transforms (best Dice={score:.5f})."
        )
    return best


def create_figure(
    *,
    b0_subject_dir: str | Path,
    no_b0_subject_dir: str | Path,
    pipelines: Sequence[tuple[str, str]],
    metabolites: Mapping[str, str],
    superior_slice: int,
    inferior_slice: int,
    vmax_percentile: float | None = 98,
    vmax_reference_pipeline: str | None = None,
    anatomy_upsampling: int = 4,
    cmap: str = "viridis",
    map_alpha: float = 0.72,
    localizer_orientation: str = "coronal",
    font_scale: float = 1.0,
    figsize: tuple[float, float] = (13, 9.4),
    selected_voxels: Mapping[str, Sequence[int]] | None = None,
    stack_slices: bool = False,
):
    """Return the figure and a dictionary of geometry/QC information."""
    b0_subject_dir = Path(b0_subject_dir).expanduser().resolve()
    no_b0_subject_dir = Path(no_b0_subject_dir).expanduser().resolve()
    source_maps = no_b0_subject_dir / "OriginalData/maps"
    mask_path = source_maps / "mask.nii.gz"
    anatomy_path = source_maps / "magnitude.nii.gz"
    if not mask_path.is_file() or not anatomy_path.is_file():
        raise FileNotFoundError(
            f"Expected mask.nii.gz and magnitude.nii.gz in {source_maps}"
        )

    mask_image = nib.load(mask_path)
    mask = np.asarray(mask_image.dataobj).squeeze() > 0
    if mask.ndim != 3:
        raise ValueError(f"Brain mask is not 3-D: {mask.shape}")
    for name, index in (("superior_slice", superior_slice),
                        ("inferior_slice", inferior_slice)):
        if not 0 <= index < mask.shape[2]:
            raise IndexError(f"{name}={index}; valid range is 0..{mask.shape[2] - 1}")
    if superior_slice <= inferior_slice:
        raise ValueError("superior_slice must be greater than inferior_slice")

    array_mask_path = b0_subject_dir / "masks/brain_mask.npy"
    if not array_mask_path.is_file():
        raise FileNotFoundError(array_mask_path)
    array_mask = np.asarray(np.load(array_mask_path)).squeeze() > 0
    orientation_dice, orientation_transform = _resolve_array_orientation(
        array_mask, mask
    )
    transformed_voxels = {}
    for label, index in (selected_voxels or {}).items():
        index = tuple(int(value) for value in index)
        if len(index) != 3 or any(
            value < 0 or value >= array_mask.shape[axis]
            for axis, value in enumerate(index)
        ):
            raise IndexError(
                f"Selected voxel {label!r}={index} is outside {array_mask.shape}"
            )
        transformed_voxels[str(label)] = _transform_index(
            index, array_mask.shape, **orientation_transform
        )

    fine_shape, fine_affine = _fine_reference(mask_image, anatomy_upsampling)
    anatomy_image = resample_from_to(
        nib.load(anatomy_path), (fine_shape, fine_affine), order=1
    )
    anatomy = np.asarray(anatomy_image.dataobj, dtype=np.float32)
    anatomy_values = anatomy[np.isfinite(anatomy) & (anatomy > 0)]
    anatomy_vmax = float(np.percentile(anatomy_values, 99.5))

    map_dirs = [(label, _map_dir(b0_subject_dir / folder))
                for label, folder in pipelines]
    vmax_reference_index = None
    if vmax_reference_pipeline is not None:
        matches = [
            index
            for index, (label, folder) in enumerate(pipelines)
            if vmax_reference_pipeline in {label, folder}
        ]
        if len(matches) != 1:
            raise ValueError(
                "vmax_reference_pipeline must uniquely match a pipeline label "
                f"or folder; got {vmax_reference_pipeline!r}"
            )
        vmax_reference_index = matches[0]
    all_maps: dict[str, list[np.ndarray]] = {}
    for display_name, file_name in metabolites.items():
        maps = []
        for _, directory in map_dirs:
            path = directory / f"{file_name}_amp_map.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(path)
            data = np.asarray(nib.load(path).dataobj, dtype=np.float32).squeeze()
            data = _transform_spatial(data, **orientation_transform)
            if data.shape != mask.shape:
                raise ValueError(
                    f"{path}: oriented shape {data.shape} != mask {mask.shape}"
                )
            maps.append(data)
        all_maps[display_name] = maps

    n_methods = len(pipelines)
    if n_methods < 1:
        raise ValueError("At least one pipeline is required")
    if not 0 < map_alpha <= 1:
        raise ValueError("map_alpha must be in (0, 1]")
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than zero")
    localizer_orientation = localizer_orientation.lower()
    if localizer_orientation not in {"coronal", "sagittal"}:
        raise ValueError(
            "localizer_orientation must be 'coronal' or 'sagittal'"
        )
    if stack_slices:
        figure = plt.figure(figsize=figsize, constrained_layout=False)
        # Keep the overall figure height stable while adding compact headers
        # that visually separate the Slice A and Slice B map blocks.
        n_rows = 4 + 2 * len(metabolites)
        map_row_height = 0.88
        grid = figure.add_gridspec(
            n_rows, n_methods + 2,
            height_ratios=(
                [1.10, 0.22, 0.13]
                + [map_row_height] * len(metabolites)
                + [0.17]
                + [map_row_height] * len(metabolites)
            ),
            width_ratios=[0.001] + [1] * n_methods + [0.055],
            wspace=0.001,
            hspace=0.012,
        )
        figure.subplots_adjust(
            left=0.055, right=0.945, bottom=0.025, top=0.975,
            wspace=0.001, hspace=0.012,
        )

        localizer_axis = figure.add_subplot(grid[0, n_methods // 2 + 1])
        if localizer_orientation == "coronal":
            localizer = anatomy[:, anatomy.shape[1] // 2, :].T
        else:
            localizer = anatomy[anatomy.shape[0] // 2, :, :].T
        localizer_axis.imshow(
            localizer, origin="lower", cmap="gray", vmin=0,
            vmax=anatomy_vmax, interpolation="nearest"
        )
        a_fine = _fine_index(superior_slice, anatomy_upsampling)
        b_fine = _fine_index(inferior_slice, anatomy_upsampling)
        for position, label, color in ((a_fine, "A", "#e41a1c"),
                                       (b_fine, "B", "#377eb8")):
            localizer_axis.axhline(position, color=color, linewidth=1.3)
            localizer_axis.text(
                0.03, position / anatomy.shape[2], label,
                color=color, weight="bold", transform=localizer_axis.transAxes,
                ha="left", va="bottom", fontsize=9,
            )
        localizer_axis.set_title(
            f"{localizer_orientation.capitalize()}\nlocalizer", fontsize=10
        )
        localizer_axis.axis("off")

        anatomy_axes = {}
        anatomy_columns = (1, n_methods)
        for column, (slice_index, label) in zip(
            anatomy_columns,
            ((superior_slice, "A"), (inferior_slice, "B")),
        ):
            axis = figure.add_subplot(grid[0, column])
            fine_z = _fine_index(slice_index, anatomy_upsampling)
            axis.imshow(anatomy[:, :, fine_z].T, origin="lower", cmap="gray",
                        vmin=0, vmax=anatomy_vmax, interpolation="nearest")
            axis.set_title(label)
            axis.axis("off")
            anatomy_axes[slice_index] = axis

        for marker_label, (x, y, z) in transformed_voxels.items():
            axis = anatomy_axes.get(z)
            if axis is None:
                continue
            axis.text(
                _fine_index(x, anatomy_upsampling),
                _fine_index(y, anatomy_upsampling),
                marker_label,
                ha="center", va="center", color="white", weight="bold",
                fontsize=8.5, zorder=10,
                bbox={
                    "boxstyle": "circle,pad=0.22",
                    "facecolor": "#d62728",
                    "edgecolor": "white",
                    "linewidth": 1.0,
                },
            )

        for method_index, (method_label, _) in enumerate(pipelines):
            axis = figure.add_subplot(grid[1, method_index + 1])
            axis.text(0.5, 0.5, method_label, ha="center", va="center",
                      weight="bold")
            axis.axis("off")

        slice_header_rows = (2, 3 + len(metabolites))
        for header_index, (header_row, slice_label) in enumerate(
            zip(slice_header_rows, ("A", "B"))
        ):
            header_axis = figure.add_subplot(
                grid[header_row, 1:n_methods + 1]
            )
            header_axis.set_xlim(0, 1)
            header_axis.set_ylim(0, 1)
            if header_index == 1:
                header_axis.axhline(
                    0.90,
                    color="0.45",
                    linewidth=0.8,
                    linestyle=(0, (4, 3)),
                    clip_on=False,
                )
            header_axis.text(
                0.0, 0.38, f"Slice {slice_label}",
                ha="left", va="center", weight="bold", fontsize=10,
            )
            header_axis.axis("off")

        for slice_block, (slice_index, slice_label) in enumerate(
            ((superior_slice, "A"), (inferior_slice, "B"))
        ):
            for metabolite_index, (metabolite, maps) in enumerate(all_maps.items()):
                block_start = (
                    3 if slice_block == 0 else 4 + len(metabolites)
                )
                row = block_start + metabolite_index
                scale_maps = (
                    maps if vmax_reference_index is None
                    else [maps[vmax_reference_index]]
                )
                vmax = _positive_limit(scale_maps, vmax_percentile)
                label_axis = figure.add_subplot(grid[row, 0])
                label_axis.axis("off")
                fine_z = _fine_index(slice_index, anatomy_upsampling)
                anatomy_slice = anatomy[:, :, fine_z].T
                image = None
                for method_index, array in enumerate(maps):
                    axis = figure.add_subplot(grid[row, method_index + 1])
                    axis.imshow(
                        anatomy_slice, origin="lower", cmap="gray", vmin=0,
                        vmax=anatomy_vmax, interpolation="nearest"
                    )
                    shown = np.ma.masked_where(
                        (~mask[:, :, slice_index])
                        | ~np.isfinite(array[:, :, slice_index]),
                        array[:, :, slice_index],
                    )
                    image = axis.imshow(
                        shown.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                        interpolation="nearest", alpha=map_alpha,
                        extent=(-0.5, anatomy.shape[0] - 0.5,
                                -0.5, anatomy.shape[1] - 0.5),
                    )
                    if method_index == 0:
                        axis.text(
                            -0.108, 0.5, metabolite,
                            ha="center", va="center", rotation=90,
                            weight="bold", transform=axis.transAxes,
                            clip_on=False,
                        )
                    axis.axis("off")
                colorbar_axis = figure.add_subplot(grid[row, -1])
                colorbar = figure.colorbar(
                    image,
                    cax=colorbar_axis,
                    label=f"{metabolite} amplitude [a.u.]",
                )
                colorbar.locator = MaxNLocator(nbins=2)
                colorbar.update_ticks()
                colorbar.ax.yaxis.label.set_size(8)
                colorbar.ax.tick_params(labelsize=7)

        if font_scale != 1.0:
            for text_artist in figure.findobj(match=Text):
                text_artist.set_fontsize(text_artist.get_fontsize() * font_scale)

        qc = {
            "mask_path": mask_path,
            "anatomy_path": anatomy_path,
            "mask_shape": mask.shape,
            "mask_affine": mask_image.affine.copy(),
            "anatomy_shape_original": nib.load(anatomy_path).shape,
            "anatomy_shape_resampled": anatomy.shape,
            "map_directories": dict(map_dirs),
            "array_orientation_transform": orientation_transform,
            "array_orientation_dice": orientation_dice,
            "selected_voxels_array": dict(selected_voxels or {}),
            "selected_voxels_oriented": transformed_voxels,
            "selected_voxels_not_on_displayed_slices": {
                label: index for label, index in transformed_voxels.items()
                if index[2] not in anatomy_axes
            },
        }
        return figure, qc

    n_map_columns = 2 * n_methods
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    grid = figure.add_gridspec(
        2 + len(metabolites), n_map_columns + 2,
        # Extra room belongs exclusively to the anatomical row. The method
        # headings retain their compact band directly above the maps.
        height_ratios=[1.10, 0.14] + [1] * len(metabolites),
        width_ratios=[0.16] + [1] * n_map_columns + [0.055],
        wspace=0.02,
        hspace=0.02,
    )
    figure.set_constrained_layout_pads(
        w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01
    )

    # High-resolution T1 panels, resliced into the oblique MRSI orientation.
    # A and B remain centered above their method groups; the localizer uses
    # the otherwise empty space between both groups.
    localizer_grid = grid[0, n_methods:n_methods + 2].subgridspec(
        2, 1, height_ratios=[1, 0.10], hspace=0
    )
    localizer_axis = figure.add_subplot(localizer_grid[0])
    if localizer_orientation == "coronal":
        localizer = anatomy[:, anatomy.shape[1] // 2, :].T
    else:
        localizer = anatomy[anatomy.shape[0] // 2, :, :].T
    localizer_axis.imshow(
        localizer, origin="lower", cmap="gray", vmin=0,
        vmax=anatomy_vmax, interpolation="nearest"
    )
    a_fine = _fine_index(superior_slice, anatomy_upsampling)
    b_fine = _fine_index(inferior_slice, anatomy_upsampling)
    for position, label, color in ((a_fine, "A", "#e41a1c"),
                                   (b_fine, "B", "#377eb8")):
        localizer_axis.axhline(position, color=color, linewidth=1.3)
        localizer_axis.text(0.03, position / anatomy.shape[2], label,
                            color=color, weight="bold", transform=localizer_axis.transAxes,
                            ha="left", va="bottom", fontsize=9)
    localizer_axis.set_title(
        f"{localizer_orientation.capitalize()}\nlocalizer", fontsize=10
    )
    localizer_axis.axis("off")

    anatomy_axes = {}
    for group_index, (slice_index, label) in enumerate(
        ((superior_slice, "A: superior slice"),
         (inferior_slice, "B: inferior slice"))
    ):
        # With an even method count, use the left middle column for A and the
        # right middle column for B. This keeps both slices symmetric around
        # the two-column localizer without making either T1 panel oversized.
        center_column = (
            1 + (n_methods - 1) // 2
            if group_index == 0
            else 1 + n_methods + n_methods // 2
        )
        anatomy_grid = grid[0, center_column].subgridspec(
            2, 1, height_ratios=[1, 0.10], hspace=0
        )
        axis = figure.add_subplot(anatomy_grid[0])
        fine_z = _fine_index(slice_index, anatomy_upsampling)
        axis.imshow(anatomy[:, :, fine_z].T, origin="lower", cmap="gray",
                    vmin=0, vmax=anatomy_vmax, interpolation="nearest")
        axis.set_title(label)
        axis.axis("off")
        anatomy_axes[slice_index] = axis

    # Selected indices originate in the NumPy/MRSI array space. Apply the
    # same lossless orientation transform used for the maps, then place the
    # marker at the corresponding high-resolution voxel center.
    for marker_label, (x, y, z) in transformed_voxels.items():
        axis = anatomy_axes.get(z)
        if axis is None:
            continue
        axis.text(
            _fine_index(x, anatomy_upsampling),
            _fine_index(y, anatomy_upsampling),
            marker_label,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=8.5,
            zorder=10,
            bbox={
                "boxstyle": "circle,pad=0.22",
                "facecolor": "#d62728",
                "edgecolor": "white",
                "linewidth": 1.0,
            },
        )

    # Method headings for both axial slices.
    for slice_group, slice_label in enumerate(("A", "B")):
        for method_index, (method_label, _) in enumerate(pipelines):
            column = slice_group * n_methods + method_index
            axis = figure.add_subplot(grid[1, column + 1])
            axis.text(0.5, 0.5, method_label, ha="center", va="center", weight="bold")
            axis.axis("off")

    for row_offset, (metabolite, maps) in enumerate(all_maps.items()):
        row = 2 + row_offset
        scale_maps = (
            maps if vmax_reference_index is None
            else [maps[vmax_reference_index]]
        )
        vmax = _positive_limit(scale_maps, vmax_percentile)
        image = None
        label_axis = figure.add_subplot(grid[row, 0])
        label_axis.text(
            0.5, 0.5, metabolite, ha="center", va="center",
            rotation=90, weight="bold"
        )
        label_axis.axis("off")
        for slice_group, slice_index in enumerate((superior_slice, inferior_slice)):
            fine_z = _fine_index(slice_index, anatomy_upsampling)
            anatomy_slice = anatomy[:, :, fine_z].T
            for method_index, array in enumerate(maps):
                column = slice_group * n_methods + method_index
                axis = figure.add_subplot(grid[row, column + 1])
                axis.imshow(
                    anatomy_slice, origin="lower", cmap="gray", vmin=0,
                    vmax=anatomy_vmax, interpolation="nearest"
                )
                shown = np.ma.masked_where(
                    (~mask[:, :, slice_index]) | ~np.isfinite(array[:, :, slice_index]),
                    array[:, :, slice_index],
                )
                image = axis.imshow(shown.T, origin="lower", cmap=cmap, vmin=0,
                                    vmax=vmax, interpolation="nearest", alpha=map_alpha,
                                    extent=(-0.5, anatomy.shape[0] - 0.5,
                                            -0.5, anatomy.shape[1] - 0.5))
                axis.axis("off")
        colorbar_axis = figure.add_subplot(grid[row, -1])
        colorbar = figure.colorbar(
            image, cax=colorbar_axis, label="Amplitude [a.u.]"
        )
        colorbar.locator = MaxNLocator(nbins=2)
        colorbar.update_ticks()

    if font_scale != 1.0:
        for text_artist in figure.findobj(match=Text):
            text_artist.set_fontsize(text_artist.get_fontsize() * font_scale)

    qc = {
        "mask_path": mask_path,
        "anatomy_path": anatomy_path,
        "mask_shape": mask.shape,
        "mask_affine": mask_image.affine.copy(),
        "anatomy_shape_original": nib.load(anatomy_path).shape,
        "anatomy_shape_resampled": anatomy.shape,
        "map_directories": dict(map_dirs),
        "array_orientation_transform": orientation_transform,
        "array_orientation_dice": orientation_dice,
        "selected_voxels_array": dict(selected_voxels or {}),
        "selected_voxels_oriented": transformed_voxels,
        "selected_voxels_not_on_displayed_slices": {
            label: index for label, index in transformed_voxels.items()
            if index[2] not in anatomy_axes
        },
    }
    return figure, qc
