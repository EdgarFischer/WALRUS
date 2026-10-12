"""Interactive magnitude/spectrum viewer that stores selected MRSI indices."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import yaml


def _method_style(label: str, index: int) -> dict:
    """Use Full WALRUS as the prominent reference curve."""
    normalized = label.lower()
    if normalized in {"full walrus", "final walrus"}:
        return {
            "color": "black", "linestyle": "-", "linewidth": 1.55,
            "alpha": 0.90, "zorder": 4,
        }
    if (
        normalized.startswith("u20")
        or normalized == "prior-free u-net"
        or normalized.startswith("u-net (heuristic")
    ):
        return {
            "color": "#0072B2", "linestyle": "-", "linewidth": 1.1,
            "alpha": 0.70, "zorder": 1,
        }
    if (
        normalized.startswith("y16")
        or normalized == "y-net + l2 prior"
        or normalized.startswith("u-net (calibrated")
    ):
        return {
            "color": "#E69F00", "linestyle": "--", "linewidth": 1.1,
            "alpha": 0.70, "zorder": 2,
        }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ("-", "--", ":", "-.")
    return {
        "color": colors[index % len(colors)],
        "linestyle": linestyles[index % len(linestyles)],
        "linewidth": 1.1,
        "alpha": 0.70,
        "zorder": index + 1,
    }


def save_selected_voxels(
    path: str | Path,
    *,
    scanner: str,
    subject: str,
    voxels: Sequence[Sequence[int]],
) -> Path:
    """Save ordered NumPy/MRSI voxel indices as a small YAML manifest."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {"label": str(index + 1), "index": [int(v) for v in voxel]}
        for index, voxel in enumerate(voxels)
    ]
    output.write_text(
        yaml.safe_dump(
            {
                "scanner": str(scanner),
                "subject": str(subject),
                "coordinate_space": "numpy_mrsi_xyz",
                "voxels": entries,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output


def load_selected_voxels(path: str | Path) -> dict[str, tuple[int, int, int]]:
    """Load ``label -> (x, y, z)`` from a selection YAML file."""
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("coordinate_space") != "numpy_mrsi_xyz":
        raise ValueError(f"Invalid spectrum-selection file: {source}")
    result = {}
    for entry in raw.get("voxels", []):
        label = str(entry["label"])
        index = tuple(int(value) for value in entry["index"])
        if len(index) != 3:
            raise ValueError(f"{source}: voxel {label!r} is not a 3-D index")
        result[label] = index
    return result


def create_selected_spectra_figure(
    *,
    subject_dir: str | Path,
    selection_path: str | Path,
    cleaned_files: Mapping[str, str],
    ppm_axis: np.ndarray,
    ppm_range: tuple[float, float] | None = (0.5, 4.2),
    figsize: tuple[float, float] | None = None,
    font_scale: float = 1.0,
):
    """Create raw/cleaned rows for all saved voxel positions."""
    subject = Path(subject_dir).expanduser().resolve()
    original = subject / "OriginalData"
    raw = np.load(original / "data.npy", mmap_mode="r")
    cleaned = {
        label: np.load(original / filename, mmap_mode="r")
        for label, filename in cleaned_files.items()
    }
    selected = load_selected_voxels(selection_path)
    if not selected:
        raise ValueError("No voxels are stored in the selection file")
    if font_scale <= 0:
        raise ValueError("font_scale must be greater than zero")
    ppm = np.asarray(ppm_axis, dtype=float)
    if ppm.shape != (raw.shape[-1],):
        raise ValueError("ppm_axis length differs from the FID length")
    if ppm_range is None:
        visible = np.ones(ppm.shape, dtype=bool)
    else:
        low, high = sorted(ppm_range)
        visible = (ppm >= low) & (ppm <= high)
    x_values = ppm[visible]
    n_voxels = len(selected)
    figure, axes = plt.subplots(
        2, n_voxels,
        figsize=figsize or (2.55 * n_voxels, 4.2),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )
    for column, (label, voxel) in enumerate(selected.items()):
        raw_spectrum = SpectrumVoxelSelector._spectrum(raw, voxel)[visible]
        axes[0, column].plot(x_values, raw_spectrum, color="black", linewidth=1.1)
        axes[0, column].set_title(
            label, fontweight="bold", fontsize=12 * font_scale
        )
        for method_index, (method, data) in enumerate(cleaned.items()):
            spectrum = SpectrumVoxelSelector._spectrum(data, voxel)[visible]
            axes[1, column].plot(
                x_values, spectrum, label=method,
                **_method_style(method, method_index),
            )
        for row in range(2):
            axis = axes[row, column]
            axis.set_xlim(x_values.max(), x_values.min())
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=9 * font_scale)
        axes[1, column].set_xlabel("ppm", fontsize=10 * font_scale)
    axes[0, 0].set_ylabel(
        "Raw input\nAmplitude [a.u.]", fontsize=10 * font_scale
    )
    axes[1, 0].set_ylabel(
        "Nuisance removed\nAmplitude [a.u.]", fontsize=10 * font_scale
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="outside lower center", ncol=len(labels),
        frameon=False, fontsize=10 * font_scale,
    )
    return figure


class SpectrumVoxelSelector:
    """Inspect disk-backed FIDs and select up to ``max_voxels`` positions.

    Mouse click selects the voxel for inspection. Press ``a`` to add it,
    Backspace to remove the last selection, ``c`` to clear, and ``s`` to save.
    """

    def __init__(
        self,
        *,
        subject_dir: str | Path,
        scanner: str,
        cleaned_files: Mapping[str, str],
        output_path: str | Path,
        ppm_axis: np.ndarray,
        allowed_slices: Sequence[int] | None = None,
        ppm_range: tuple[float, float] | None = (0.5, 4.2),
        max_voxels: int = 4,
        z_init: int | None = None,
    ):
        self.subject_dir = Path(subject_dir).expanduser().resolve()
        self.scanner = str(scanner)
        self.output_path = Path(output_path).expanduser().resolve()
        original = self.subject_dir / "OriginalData"
        self.magnitude = np.asarray(np.load(original / "magnitude.npy")).squeeze()
        self.brain_mask = np.asarray(
            np.load(self.subject_dir / "masks/brain_mask.npy")
        ).squeeze() > 0
        self.raw = np.load(original / "data.npy", mmap_mode="r")
        self.cleaned = {
            label: np.load(original / filename, mmap_mode="r")
            for label, filename in cleaned_files.items()
        }
        if self.magnitude.shape != self.brain_mask.shape:
            raise ValueError("Magnitude and brain-mask shapes differ")
        expected = self.magnitude.shape + (self.raw.shape[-1],)
        for label, data in {"Raw": self.raw, **self.cleaned}.items():
            if data.shape != expected:
                raise ValueError(f"{label}: shape {data.shape} != {expected}")
        self.ppm = np.asarray(ppm_axis, dtype=float)
        if self.ppm.shape != (self.raw.shape[-1],):
            raise ValueError("ppm_axis length differs from the FID length")
        if ppm_range is None:
            self.visible = np.ones(self.ppm.shape, dtype=bool)
        else:
            low, high = sorted(ppm_range)
            self.visible = (self.ppm >= low) & (self.ppm <= high)
        self.allowed_slices = (
            None if allowed_slices is None else {int(z) for z in allowed_slices}
        )
        self.max_voxels = int(max_voxels)
        self.selected: list[tuple[int, int, int]] = []
        initial_z = int(z_init if z_init is not None else self.magnitude.shape[2] // 2)
        self.current = [self.magnitude.shape[0] // 2,
                        self.magnitude.shape[1] // 2, initial_z]
        self._build_figure()

    @staticmethod
    def _spectrum(data, index):
        fid = np.asarray(data[index[0], index[1], index[2], :])
        return np.real(np.fft.fftshift(np.fft.fft(fid)))

    def _build_figure(self):
        self.figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        self.image_axis, self.raw_axis, self.cleaned_axis = axes
        plt.subplots_adjust(bottom=0.23, wspace=0.28)
        z = self.current[2]
        self.image = self.image_axis.imshow(
            self.magnitude[:, :, z].T, origin="lower", cmap="gray"
        )
        self.cursor, = self.image_axis.plot(
            self.current[0], self.current[1], marker="+", color="red", markersize=10
        )
        self.image_axis.set_title(f"Magnitude, z={z}")
        self.image_axis.set_xlabel("x")
        self.image_axis.set_ylabel("y")
        self.raw_line, = self.raw_axis.plot([], [], color="black", linewidth=1.2)
        self.raw_axis.set_title("Raw input")
        self.cleaned_lines = {}
        for method_index, label in enumerate(self.cleaned):
            line, = self.cleaned_axis.plot(
                [], [], label=label, **_method_style(label, method_index),
            )
            self.cleaned_lines[label] = line
        self.cleaned_axis.set_title("Nuisance removed")
        self.cleaned_axis.legend(frameon=False)
        for axis in (self.raw_axis, self.cleaned_axis):
            axis.set_xlabel("ppm")
            axis.set_xlim(self.ppm[self.visible].max(), self.ppm[self.visible].min())
            axis.spines[["top", "right"]].set_visible(False)
        self.raw_axis.set_ylabel("Amplitude [a.u.]")
        self.cleaned_axis.set_ylabel("Amplitude [a.u.]")
        slider_axis = self.figure.add_axes([0.10, 0.10, 0.39, 0.035])
        self.slider = Slider(
            slider_axis, "z", 0, self.magnitude.shape[2] - 1,
            valinit=z, valstep=1,
        )
        button_specs = (
            ("Add", 0.54, self._add_current),
            ("Undo", 0.64, self._undo),
            ("Clear", 0.74, self._clear),
            ("Save", 0.84, self._save_selection),
        )
        self.buttons = []
        for label, left, callback in button_specs:
            button = Button(self.figure.add_axes([left, 0.075, 0.08, 0.07]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.status = self.figure.text(
            0.02, 0.01,
            "Click the magnitude image to inspect a voxel, then use the buttons.",
            fontsize=9,
        )
        self.selection_artists = []
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.slider.on_changed(self._on_slice)
        self._redraw_spectra()

    def _set_status(self, message):
        self.status.set_text(message)
        self.figure.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.image_axis or event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x < self.magnitude.shape[0] and 0 <= y < self.magnitude.shape[1]):
            return
        self.current[:2] = [x, y]
        self.cursor.set_data([x], [y])
        self._redraw_spectra()

    def _on_slice(self, _):
        self.current[2] = int(self.slider.val)
        z = self.current[2]
        self.image.set_data(self.magnitude[:, :, z].T)
        self.image_axis.set_title(f"Magnitude, z={z}")
        self._redraw_selections()
        self._redraw_spectra()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "a":
            self._add_current()
        elif key in {"backspace", "delete"}:
            self._undo()
        elif key == "c":
            self._clear()
        elif key == "s":
            self._save_selection()
        else:
            return

    def _add_current(self, _event=None):
        voxel = tuple(self.current)
        if self.allowed_slices is not None and voxel[2] not in self.allowed_slices:
            self._set_status(f"z={voxel[2]} is not one of {sorted(self.allowed_slices)}")
            return
        if not self.brain_mask[voxel]:
            self._set_status(f"Voxel {voxel} lies outside the brain mask")
            return
        if voxel in self.selected:
            self._set_status(f"Voxel {voxel} is already selected")
            return
        if len(self.selected) >= self.max_voxels:
            self._set_status(f"Maximum of {self.max_voxels} voxels reached")
            return
        self.selected.append(voxel)
        self._set_status(f"Added {len(self.selected)}: {voxel}")
        self._redraw_selections()

    def _undo(self, _event=None):
        if not self.selected:
            self._set_status("There is no selection to remove")
            return
        removed = self.selected.pop()
        self._set_status(f"Removed {removed}")
        self._redraw_selections()

    def _clear(self, _event=None):
        self.selected.clear()
        self._set_status("Selection cleared")
        self._redraw_selections()

    def _save_selection(self, _event=None):
        if not self.selected:
            self._set_status("Select at least one voxel before saving")
            return
        saved = self.save()
        self._set_status(f"Saved {len(self.selected)} voxels to {saved}")

    def _redraw_spectra(self):
        voxel = tuple(self.current)
        x = self.ppm[self.visible]
        self.raw_line.set_data(x, self._spectrum(self.raw, voxel)[self.visible])
        self.raw_axis.relim()
        self.raw_axis.autoscale_view(scalex=False, scaley=True)
        for label, data in self.cleaned.items():
            self.cleaned_lines[label].set_data(
                x, self._spectrum(data, voxel)[self.visible]
            )
        self.cleaned_axis.relim()
        self.cleaned_axis.autoscale_view(scalex=False, scaley=True)
        self.figure.canvas.draw_idle()

    def _redraw_selections(self):
        for artist in self.selection_artists:
            artist.remove()
        self.selection_artists.clear()
        z = self.current[2]
        for number, (x, y, selected_z) in enumerate(self.selected, start=1):
            if selected_z != z:
                continue
            artist = self.image_axis.text(
                x, y, str(number), color="white", ha="center", va="center",
                weight="bold", fontsize=9,
                bbox={"boxstyle": "circle,pad=0.22", "facecolor": "#d62728",
                      "edgecolor": "white", "linewidth": 1.0},
            )
            self.selection_artists.append(artist)
        self.figure.canvas.draw_idle()

    def save(self) -> Path:
        if not self.selected:
            raise ValueError("Select at least one voxel before saving")
        return save_selected_voxels(
            self.output_path,
            scanner=self.scanner,
            subject=str(self.subject_dir.name),
            voxels=self.selected,
        )
