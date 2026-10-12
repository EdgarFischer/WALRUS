"""Compact two-method, two-metabolite full-length map comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.text import Text
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from walinet.paper_figures.metabolite_map_figure.figure import (
    _fine_index,
    _fine_reference,
    _map_dir,
    _positive_limit,
    _resolve_array_orientation,
    _transform_spatial,
)


def create_two_method_full_length_figure(
    *,
    b0_subject_dir: str | Path,
    no_b0_subject_dir: str | Path,
    pipelines: Sequence[tuple[str, str]],
    metabolites: Mapping[str, str],
    superior_slice: int,
    inferior_slice: int,
    vmax_reference_pipeline: str,
    vmax_percentile: float | None = 97.5,
    anatomy_upsampling: int = 4,
    cmap: str = "viridis",
    map_alpha: float = 1.0,
    localizer_orientation: str = "sagittal",
    font_scale: float = 1.0,
    figsize: tuple[float, float] = (16.5, 4.6),
):
    """Return a compact figure with anatomy on top and eight maps below."""
    if len(pipelines) != 2:
        raise ValueError("This compact layout expects exactly two pipelines")
    if len(metabolites) != 2:
        raise ValueError("This compact layout expects exactly two metabolites")

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
    for name, index in (("superior_slice", superior_slice),
                        ("inferior_slice", inferior_slice)):
        if not 0 <= index < mask.shape[2]:
            raise IndexError(f"{name}={index}; valid range is 0..{mask.shape[2] - 1}")

    array_mask_path = b0_subject_dir / "masks/brain_mask.npy"
    array_mask = np.asarray(np.load(array_mask_path)).squeeze() > 0
    orientation_dice, orientation_transform = _resolve_array_orientation(
        array_mask, mask
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

    figure = plt.figure(figsize=figsize, constrained_layout=False)

    method_labels = [label for label, _ in pipelines]
    metabolite_names = list(metabolites)
    slice_labels = ("A", "B")
    slice_indices = (superior_slice, inferior_slice)

    left = 0.035
    right = 0.985
    gap = 0.015
    map_bottom = 0.045
    map_height = 0.60
    header_top = 0.965
    method_y = 0.905
    metabolite_y = 0.795
    slice_y = 0.690
    map_width = (right - left - 7 * gap) / 8
    centers = [left + index * (map_width + gap) + map_width / 2 for index in range(8)]

    def _figure_text(x, y, text, *, fontsize, weight="normal"):
        figure.text(
            x, y, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
        )

    _figure_text(
        np.mean(centers[:4]), method_y, method_labels[0],
        fontsize=13.5, weight="bold",
    )
    _figure_text(
        np.mean(centers[4:]), method_y, method_labels[1],
        fontsize=13.5, weight="bold",
    )
    for base_index in (0, 4):
        _figure_text(
            np.mean(centers[base_index:base_index + 2]),
            metabolite_y, metabolite_names[0],
            fontsize=12.5, weight="bold",
        )
        _figure_text(
            np.mean(centers[base_index + 2:base_index + 4]),
            metabolite_y, metabolite_names[1],
            fontsize=12.5, weight="bold",
        )
        for pair_base in (base_index, base_index + 2):
            for offset, slice_label in enumerate(slice_labels):
                _figure_text(
                    centers[pair_base + offset], slice_y, slice_label,
                    fontsize=11.5, weight="bold",
                )

    # Columns: No suppression tNAA A/B, No suppression Glx A/B,
    # WALRUS tNAA A/B, WALRUS Glx A/B. Positions are manually fixed so every
    # map has exactly the same width and spacing, independent of text extents.
    map_specs = []
    for method_index, base_column in ((0, 0), (1, 4)):
        for metabolite_offset, metabolite in enumerate(metabolite_names):
            for slice_offset, slice_index in enumerate(slice_indices):
                column = base_column + 2 * metabolite_offset + slice_offset
                map_specs.append((column, metabolite, slice_index, method_index))

    for column, metabolite, slice_index, method_index in map_specs:
        maps = all_maps[metabolite]
        vmax = _positive_limit([maps[vmax_reference_index]], vmax_percentile)
        fine_z = _fine_index(slice_index, anatomy_upsampling)
        anatomy_slice = anatomy[:, :, fine_z].T
        axis = figure.add_axes([
            left + column * (map_width + gap),
            map_bottom,
            map_width,
            map_height,
        ])
        axis.imshow(anatomy_slice, origin="lower", cmap="gray", vmin=0,
                    vmax=anatomy_vmax, interpolation="nearest")
        array = maps[method_index]
        shown = np.ma.masked_where(
            (~mask[:, :, slice_index]) | ~np.isfinite(array[:, :, slice_index]),
            array[:, :, slice_index],
        )
        axis.imshow(
            shown.T, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
            interpolation="nearest", alpha=map_alpha,
            extent=(-0.5, anatomy.shape[0] - 0.5,
                    -0.5, anatomy.shape[1] - 0.5),
        )
        axis.axis("off")

    if font_scale != 1.0:
        for text_artist in figure.findobj(match=Text):
            text_artist.set_fontsize(text_artist.get_fontsize() * font_scale)

    qc = {
        "mask_path": mask_path,
        "anatomy_path": anatomy_path,
        "mask_shape": mask.shape,
        "anatomy_shape_original": nib.load(anatomy_path).shape,
        "anatomy_shape_resampled": anatomy.shape,
        "map_directories": dict(map_dirs),
        "array_orientation_transform": orientation_transform,
        "array_orientation_dice": orientation_dice,
    }
    return figure, qc
