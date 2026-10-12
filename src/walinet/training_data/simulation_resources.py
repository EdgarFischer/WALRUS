# src/walinet/training_data/simulation_resources.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from walinet.config.schema import TrainConfig
from walinet.config.schema_simulation import SimulationConfig


def _subject_names_match(
    *,
    requested_subject: str,
    stored_subject: str,
) -> bool:
    """Match current ``site/subject`` paths to legacy HDF5 subject names."""
    requested = Path(requested_subject)
    requested_normalized = requested.as_posix()

    if stored_subject == requested_normalized:
        return True

    parts = requested.parts
    if len(parts) != 2:
        return False

    site, subject = parts
    legacy_aliases = {
        subject,
        f"{subject}_{site}",
    }
    return stored_subject in legacy_aliases


@dataclass(frozen=True)
class SimulationPool:
    """
    In-memory simulation resources for one data split.

    The HDF5 resource files store water and lipid signals as FIDs.
    During loading, the FIDs are prepared to the target acquisition
    length and transformed once into fft-shifted spectra.

    The simulator therefore receives frequency-domain resources and
    does not need to transform water or lipid signals for every batch.

    Shapes
    ------
    water_spectra:
        (N_water_total, T)

    lipid_spectra:
        (N_lipid_total, T)

    water_offsets:
        (n_subjects + 1,)

    lipid_offsets:
        (n_subjects + 1,)

    native_lengths:
        (n_subjects,)

    lipid_projection_operators:
        (n_subjects, T, T), or None
    """

    subject_names: tuple[str, ...]

    water_spectra: torch.Tensor
    lipid_spectra: torch.Tensor

    water_offsets: torch.Tensor
    lipid_offsets: torch.Tensor

    native_lengths: torch.Tensor

    lipid_projection_operators: torch.Tensor | None

    bandwidth_hz: float
    n_timepoints: int

    @property
    def n_subjects(self) -> int:
        return len(self.subject_names)

    @property
    def n_water_spectra(self) -> int:
        return int(
            self.water_spectra.shape[0]
        )

    @property
    def n_lipid_spectra(self) -> int:
        return int(
            self.lipid_spectra.shape[0]
        )

    @property
    def water_counts(self) -> torch.Tensor:
        """
        Number of valid water spectra for every subject.
        """
        return (
            self.water_offsets[1:]
            - self.water_offsets[:-1]
        )

    @property
    def lipid_counts(self) -> torch.Tensor:
        """
        Number of valid lipid spectra for every subject.
        """
        return (
            self.lipid_offsets[1:]
            - self.lipid_offsets[:-1]
        )

    @property
    def device(self) -> torch.device:
        return self.water_spectra.device

    def water_for_subject(
        self,
        subject_index: int,
    ) -> torch.Tensor:
        """
        Return all water spectra belonging to one subject.

        Intended mainly for debugging and notebooks.
        """
        self._validate_subject_index(
            subject_index
        )

        start = int(
            self.water_offsets[
                subject_index
            ].item()
        )

        end = int(
            self.water_offsets[
                subject_index + 1
            ].item()
        )

        return self.water_spectra[
            start:end
        ]

    def lipids_for_subject(
        self,
        subject_index: int,
    ) -> torch.Tensor:
        """
        Return all lipid spectra belonging to one subject.

        Intended mainly for debugging and notebooks.
        """
        self._validate_subject_index(
            subject_index
        )

        start = int(
            self.lipid_offsets[
                subject_index
            ].item()
        )

        end = int(
            self.lipid_offsets[
                subject_index + 1
            ].item()
        )

        return self.lipid_spectra[
            start:end
        ]

    def subject_name(
        self,
        subject_index: int,
    ) -> str:
        self._validate_subject_index(
            subject_index
        )

        return self.subject_names[
            subject_index
        ]

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SimulationPool:
        """
        Move all tensors of the pool to one device.

        The loader initially creates CPU tensors. The complete pool
        can subsequently be moved to a GPU once.
        """
        target_device = torch.device(
            device
        )

        projection_operators = (
            None
            if self.lipid_projection_operators is None
            else self.lipid_projection_operators.to(
                target_device,
                non_blocking=non_blocking,
            )
        )

        return SimulationPool(
            subject_names=self.subject_names,
            water_spectra=(
                self.water_spectra.to(
                    target_device,
                    non_blocking=non_blocking,
                )
            ),
            lipid_spectra=(
                self.lipid_spectra.to(
                    target_device,
                    non_blocking=non_blocking,
                )
            ),
            water_offsets=(
                self.water_offsets.to(
                    target_device,
                    non_blocking=non_blocking,
                )
            ),
            lipid_offsets=(
                self.lipid_offsets.to(
                    target_device,
                    non_blocking=non_blocking,
                )
            ),
            native_lengths=(
                self.native_lengths.to(
                    target_device,
                    non_blocking=non_blocking,
                )
            ),
            lipid_projection_operators=(
                projection_operators
            ),
            bandwidth_hz=self.bandwidth_hz,
            n_timepoints=self.n_timepoints,
        )

    def _validate_subject_index(
        self,
        subject_index: int,
    ) -> None:
        if not (
            0
            <= subject_index
            < self.n_subjects
        ):
            raise IndexError(
                "subject_index is out of range:\n"
                f"  subject_index: {subject_index}\n"
                f"  n_subjects:   {self.n_subjects}"
            )


@dataclass(frozen=True)
class SimulationResources:
    """
    Complete train and validation simulation resources.
    """

    train: SimulationPool
    validation: SimulationPool

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> SimulationResources:
        """
        Move both pools to one device.
        """
        return SimulationResources(
            train=self.train.to(
                device,
                non_blocking=non_blocking,
            ),
            validation=self.validation.to(
                device,
                non_blocking=non_blocking,
            ),
        )


@dataclass(frozen=True)
class _LoadedSubjectResources:
    """
    Internal frequency-domain representation of one subject.

    The data are read from the HDF5 file as FIDs but converted into
    spectra before this object is returned.
    """

    subject: str

    water_spectra: np.ndarray
    lipid_spectra: np.ndarray

    native_n_timepoints: int
    bandwidth_hz: float

    lipid_projection_operator: (
        np.ndarray | None
    )


def _read_string_attribute(
    value: object,
) -> str:
    """
    Convert an HDF5 string attribute to a normal Python string.
    """
    if isinstance(value, bytes):
        return value.decode(
            "utf-8"
        )

    return str(value)


def _require_dataset(
    h5: h5py.File | h5py.Group,
    name: str,
    *,
    resource_path: Path,
) -> h5py.Dataset:
    """
    Return a required HDF5 dataset or raise an informative error.
    """
    if name not in h5:
        raise KeyError(
            f"Required dataset {name!r} is missing in:\n"
            f"  {resource_path}"
        )

    dataset = h5[name]

    if not isinstance(
        dataset,
        h5py.Dataset,
    ):
        raise TypeError(
            f"HDF5 entry {name!r} is not a dataset in:\n"
            f"  {resource_path}"
        )

    return dataset


def _prepare_fid_pool(
    fids: np.ndarray,
    *,
    target_n_timepoints: int,
) -> np.ndarray:
    """
    Crop or zero-fill a two-dimensional FID pool.

    Shape
    -----
    input:
        (N, T_native)

    output:
        (N, T_target)

    Cropping:
        Keep the first target_n_timepoints acquired FID samples.

    Zero-filling:
        Copy native samples to the beginning and append zeros.
    """
    if fids.ndim != 2:
        raise ValueError(
            "FID pool must have shape (N, T), "
            f"but found {fids.shape}."
        )

    if target_n_timepoints <= 0:
        raise ValueError(
            "target_n_timepoints must be > 0."
        )

    native_n_timepoints = int(
        fids.shape[-1]
    )

    if (
        native_n_timepoints
        == target_n_timepoints
    ):
        return np.ascontiguousarray(
            fids,
            dtype=np.complex64,
        )

    prepared = np.zeros(
        (
            fids.shape[0],
            target_n_timepoints,
        ),
        dtype=np.complex64,
    )

    n_copy = min(
        native_n_timepoints,
        target_n_timepoints,
    )

    prepared[:, :n_copy] = np.asarray(
        fids[:, :n_copy],
        dtype=np.complex64,
    )

    return np.ascontiguousarray(
        prepared,
        dtype=np.complex64,
    )


def _fids_to_spectra(
    fids: np.ndarray,
) -> np.ndarray:
    """
    Convert a prepared FID pool into fft-shifted spectra.

    The FFT convention is identical to the convention used throughout
    the simulator:

        spectrum = fftshift(fft(fid))

    Parameters
    ----------
    fids:
        Complex array with shape (N, T).

    Returns
    -------
    np.ndarray:
        Contiguous complex64 array with shape (N, T).
    """
    if fids.ndim != 2:
        raise ValueError(
            "FID pool must have shape (N, T), "
            f"but found {fids.shape}."
        )

    if not np.iscomplexobj(
        fids
    ):
        raise TypeError(
            "FID pool must be complex-valued."
        )

    spectra = np.fft.fftshift(
        np.fft.fft(
            fids,
            axis=-1,
        ),
        axes=-1,
    )

    if not np.isfinite(
        spectra
    ).all():
        raise ValueError(
            "Generated spectra contain NaN or Inf."
        )

    return np.ascontiguousarray(
        spectra,
        dtype=np.complex64,
    )


def _validate_and_filter_pool(
    fids: np.ndarray,
    *,
    subject: str,
    resource_name: str,
) -> np.ndarray:
    """
    Validate a compact FID pool and remove identically zero rows.

    NaN or Inf values are treated as errors. Empty rows are removed
    because they do not provide useful simulation resources.
    """
    if fids.ndim != 2:
        raise ValueError(
            f"{resource_name} for {subject} must have shape "
            f"(N, T), but found {fids.shape}."
        )

    finite_rows = np.isfinite(
        fids
    ).all(
        axis=-1,
    )

    if not finite_rows.all():
        n_invalid = int(
            (~finite_rows).sum()
        )

        raise ValueError(
            f"{resource_name} for {subject} contains "
            f"{n_invalid} FID row(s) with NaN or Inf."
        )

    nonzero_rows = np.any(
        fids != 0,
        axis=-1,
    )

    n_empty = int(
        (~nonzero_rows).sum()
    )

    if n_empty > 0:
        print(
            f"[Resources] {subject}: removing "
            f"{n_empty} empty {resource_name} FID(s)."
        )

    filtered = fids[
        nonzero_rows
    ]

    if filtered.shape[0] == 0:
        raise ValueError(
            f"No valid {resource_name} FIDs remain "
            f"for subject {subject}."
        )

    return np.ascontiguousarray(
        filtered,
        dtype=np.complex64,
    )


def _load_projection_operator(
    *,
    h5: h5py.File,
    subject: str,
    resource_path: Path,
    n_timepoints: int,
) -> np.ndarray:
    """
    Load the subject-specific frequency-domain projection operator.
    """
    group_name = "lipid_projection"

    if group_name not in h5:
        raise KeyError(
            "Lipid projection is enabled, but the resource file "
            "does not contain a lipid_projection group:\n"
            f"  subject: {subject}\n"
            f"  file:    {resource_path}"
        )

    group = h5[
        group_name
    ]

    if not isinstance(
        group,
        h5py.Group,
    ):
        raise TypeError(
            "The lipid_projection entry is not an HDF5 group:\n"
            f"  {resource_path}"
        )

    dataset_name = (
        f"operator_{n_timepoints}"
    )

    operator_dataset = _require_dataset(
        group,
        dataset_name,
        resource_path=resource_path,
    )

    operator = np.asarray(
        operator_dataset[:],
        dtype=np.complex64,
    )

    expected_shape = (
        n_timepoints,
        n_timepoints,
    )

    if operator.shape != expected_shape:
        raise ValueError(
            f"Projection operator for {subject} has the wrong "
            "shape:\n"
            f"  expected: {expected_shape}\n"
            f"  found:    {operator.shape}\n"
            f"  file:     {resource_path}"
        )

    if not np.isfinite(
        operator
    ).all():
        raise ValueError(
            f"Projection operator for {subject} contains "
            "NaN or Inf."
        )

    if "domain" in group.attrs:
        domain = _read_string_attribute(
            group.attrs["domain"]
        )

        if domain != "frequency":
            raise ValueError(
                "Projection operator has unexpected domain:\n"
                f"  expected: frequency\n"
                f"  found:    {domain}\n"
                f"  file:     {resource_path}"
            )

    if "fft_shifted" in group.attrs:
        if not bool(
            group.attrs["fft_shifted"]
        ):
            raise ValueError(
                "Projection operator is not marked as "
                "fft-shifted:\n"
                f"  {resource_path}"
            )

    return np.ascontiguousarray(
        operator,
        dtype=np.complex64,
    )


def _load_subject_resources(
    *,
    subject: str,
    resource_path: Path,
    target_n_timepoints: int,
    load_projection_operator: bool,
) -> _LoadedSubjectResources:
    """
    Load and validate one subject-specific resource file.

    The HDF5 file contains FIDs. They are cropped or zero-filled and
    then converted once into fft-shifted spectra before returning.
    """
    if not resource_path.is_file():
        raise FileNotFoundError(
            "Simulation resource file not found:\n"
            f"  subject: {subject}\n"
            f"  file:    {resource_path}"
        )

    print(
        f"[Resources] Loading {subject}:"
    )
    print(
        f"  {resource_path}"
    )

    with h5py.File(
        resource_path,
        "r",
    ) as h5:
        if "domain" not in h5.attrs:
            raise ValueError(
                "Resource file has no domain attribute:\n"
                f"  {resource_path}"
            )

        domain = _read_string_attribute(
            h5.attrs["domain"]
        )

        if domain != "fid":
            raise ValueError(
                "Simulation resources must be stored in the "
                "FID domain:\n"
                f"  expected: fid\n"
                f"  found:    {domain}\n"
                f"  file:     {resource_path}"
            )

        if "subject" in h5.attrs:
            stored_subject = (
                _read_string_attribute(
                    h5.attrs["subject"]
                )
            )

            if not _subject_names_match(
                requested_subject=subject,
                stored_subject=stored_subject,
            ):
                raise ValueError(
                    "Subject name in the HDF5 file does not match "
                    "the requested subject:\n"
                    f"  requested: {subject}\n"
                    f"  stored:    {stored_subject}\n"
                    f"  file:      {resource_path}"
                )

        if "bandwidth_hz" not in h5.attrs:
            raise ValueError(
                "Resource file has no bandwidth_hz attribute:\n"
                f"  {resource_path}"
            )

        bandwidth_hz = float(
            h5.attrs["bandwidth_hz"]
        )

        if bandwidth_hz <= 0:
            raise ValueError(
                "Invalid bandwidth_hz in resource file:\n"
                f"  bandwidth: {bandwidth_hz}\n"
                f"  file:      {resource_path}"
            )

        water_dataset = _require_dataset(
            h5,
            "water_fids",
            resource_path=resource_path,
        )

        lipid_dataset = _require_dataset(
            h5,
            "lipid_fids",
            resource_path=resource_path,
        )

        brain_mask_dataset = _require_dataset(
            h5,
            "brain_mask",
            resource_path=resource_path,
        )

        water_volume = np.asarray(
            water_dataset[:],
            dtype=np.complex64,
        )

        lipid_fids = np.asarray(
            lipid_dataset[:],
            dtype=np.complex64,
        )

        brain_mask = np.asarray(
            brain_mask_dataset[:],
        ).astype(
            bool,
            copy=False,
        )

        if water_volume.ndim != 4:
            raise ValueError(
                f"water_fids for {subject} must have shape "
                "(X, Y, Z, T), but found "
                f"{water_volume.shape}."
            )

        if brain_mask.ndim != 3:
            raise ValueError(
                f"brain_mask for {subject} must have shape "
                "(X, Y, Z), but found "
                f"{brain_mask.shape}."
            )

        if (
            water_volume.shape[:-1]
            != brain_mask.shape
        ):
            raise ValueError(
                f"Water volume and brain mask do not match "
                f"for {subject}:\n"
                f"  water: {water_volume.shape[:-1]}\n"
                f"  mask:  {brain_mask.shape}"
            )

        if lipid_fids.ndim != 2:
            raise ValueError(
                f"lipid_fids for {subject} must have shape "
                f"(N, T), but found {lipid_fids.shape}."
            )

        native_n_timepoints = int(
            water_volume.shape[-1]
        )

        if (
            lipid_fids.shape[-1]
            != native_n_timepoints
        ):
            raise ValueError(
                f"Water and lipid FIDs have different native "
                f"lengths for {subject}:\n"
                f"  water: {native_n_timepoints}\n"
                f"  lipid: {lipid_fids.shape[-1]}"
            )

        if "native_n_timepoints" in h5.attrs:
            stored_native_length = int(
                h5.attrs[
                    "native_n_timepoints"
                ]
            )

            if (
                stored_native_length
                != native_n_timepoints
            ):
                raise ValueError(
                    "native_n_timepoints attribute does not "
                    f"match the data for {subject}:\n"
                    f"  attribute: {stored_native_length}\n"
                    f"  data:      {native_n_timepoints}"
                )

        if not np.isfinite(
            water_volume
        ).all():
            raise ValueError(
                f"water_fids for {subject} contains NaN or Inf."
            )

        if not np.any(
            brain_mask
        ):
            raise ValueError(
                f"brain_mask for {subject} is empty."
            )

        water_fids = np.asarray(
            water_volume[
                brain_mask
            ],
            dtype=np.complex64,
        )

        water_fids = _validate_and_filter_pool(
            water_fids,
            subject=subject,
            resource_name="water",
        )

        lipid_fids = _validate_and_filter_pool(
            lipid_fids,
            subject=subject,
            resource_name="lipid",
        )

        water_fids = _prepare_fid_pool(
            water_fids,
            target_n_timepoints=(
                target_n_timepoints
            ),
        )

        lipid_fids = _prepare_fid_pool(
            lipid_fids,
            target_n_timepoints=(
                target_n_timepoints
            ),
        )

        # Transform once during loading. From this point onward,
        # the simulator works directly in the frequency domain.
        water_spectra = _fids_to_spectra(
            water_fids
        )

        lipid_spectra = _fids_to_spectra(
            lipid_fids
        )

        projection_operator = None

        if load_projection_operator:
            projection_operator = (
                _load_projection_operator(
                    h5=h5,
                    subject=subject,
                    resource_path=resource_path,
                    n_timepoints=(
                        target_n_timepoints
                    ),
                )
            )

    print(
        f"  water spectra: {water_spectra.shape}"
    )
    print(
        f"  lipid spectra: {lipid_spectra.shape}"
    )
    print(
        f"  native length: {native_n_timepoints}"
    )
    print(
        f"  prepared length: {target_n_timepoints}"
    )

    if projection_operator is not None:
        print(
            "  projection operator: "
            f"{projection_operator.shape}"
        )

    return _LoadedSubjectResources(
        subject=subject,
        water_spectra=water_spectra,
        lipid_spectra=lipid_spectra,
        native_n_timepoints=(
            native_n_timepoints
        ),
        bandwidth_hz=bandwidth_hz,
        lipid_projection_operator=(
            projection_operator
        ),
    )


def _bandwidths_match(
    first: float,
    second: float,
) -> bool:
    return bool(
        np.isclose(
            first,
            second,
            rtol=1e-9,
            atol=1e-6,
        )
    )


def load_simulation_pool(
    *,
    base_dir: str | Path,
    subjects: list[str] | tuple[str, ...],
    resource_filename: str,
    target_n_timepoints: int,
    expected_bandwidth_hz: float,
    load_projection_operator: bool,
) -> SimulationPool:
    """
    Load and concatenate subject-specific resources for one split.

    The HDF5 files contain FIDs, but the resulting SimulationPool
    contains only frequency-domain spectra.

    The resulting tensors are initially stored on the CPU.
    """
    base_dir = Path(
        base_dir
    )

    if not subjects:
        raise ValueError(
            "subjects must not be empty."
        )

    if target_n_timepoints <= 0:
        raise ValueError(
            "target_n_timepoints must be > 0."
        )

    if expected_bandwidth_hz <= 0:
        raise ValueError(
            "expected_bandwidth_hz must be > 0."
        )

    resource_relative_path = Path(
        resource_filename
    )

    if resource_relative_path.is_absolute():
        raise ValueError(
            "resource_filename must be relative to each "
            "subject directory, but an absolute path was given:\n"
            f"  {resource_filename}"
        )

    loaded_subjects: list[
        _LoadedSubjectResources
    ] = []

    for subject in subjects:
        subject_resource_path = (
            base_dir
            / subject
            / resource_relative_path
        )

        loaded = _load_subject_resources(
            subject=subject,
            resource_path=subject_resource_path,
            target_n_timepoints=(
                target_n_timepoints
            ),
            load_projection_operator=(
                load_projection_operator
            ),
        )

        if not _bandwidths_match(
            loaded.bandwidth_hz,
            expected_bandwidth_hz,
        ):
            raise ValueError(
                f"Resource bandwidth for {subject} does not "
                "match the simulation configuration:\n"
                f"  resource:   {loaded.bandwidth_hz}\n"
                f"  simulation: {expected_bandwidth_hz}"
            )

        loaded_subjects.append(
            loaded
        )

    water_arrays: list[np.ndarray] = []
    lipid_arrays: list[np.ndarray] = []

    water_offsets = [0]
    lipid_offsets = [0]

    native_lengths: list[int] = []

    projection_operators: list[
        np.ndarray
    ] = []

    for loaded in loaded_subjects:
        water_arrays.append(
            loaded.water_spectra
        )

        lipid_arrays.append(
            loaded.lipid_spectra
        )

        water_offsets.append(
            water_offsets[-1]
            + loaded.water_spectra.shape[0]
        )

        lipid_offsets.append(
            lipid_offsets[-1]
            + loaded.lipid_spectra.shape[0]
        )

        native_lengths.append(
            loaded.native_n_timepoints
        )

        if load_projection_operator:
            if (
                loaded.lipid_projection_operator
                is None
            ):
                raise RuntimeError(
                    "Projection-operator loading was requested, "
                    f"but no operator was loaded for {loaded.subject}."
                )

            projection_operators.append(
                loaded.lipid_projection_operator
            )

    water_array = np.ascontiguousarray(
        np.concatenate(
            water_arrays,
            axis=0,
        ),
        dtype=np.complex64,
    )

    lipid_array = np.ascontiguousarray(
        np.concatenate(
            lipid_arrays,
            axis=0,
        ),
        dtype=np.complex64,
    )

    water_offsets_array = np.asarray(
        water_offsets,
        dtype=np.int64,
    )

    lipid_offsets_array = np.asarray(
        lipid_offsets,
        dtype=np.int64,
    )

    native_lengths_array = np.asarray(
        native_lengths,
        dtype=np.int64,
    )

    if load_projection_operator:
        projection_array = np.ascontiguousarray(
            np.stack(
                projection_operators,
                axis=0,
            ),
            dtype=np.complex64,
        )

        projection_tensor: (
            torch.Tensor | None
        ) = torch.from_numpy(
            projection_array
        )

    else:
        projection_tensor = None

    pool = SimulationPool(
        subject_names=tuple(
            subjects
        ),
        water_spectra=torch.from_numpy(
            water_array
        ),
        lipid_spectra=torch.from_numpy(
            lipid_array
        ),
        water_offsets=torch.from_numpy(
            water_offsets_array
        ),
        lipid_offsets=torch.from_numpy(
            lipid_offsets_array
        ),
        native_lengths=torch.from_numpy(
            native_lengths_array
        ),
        lipid_projection_operators=(
            projection_tensor
        ),
        bandwidth_hz=float(
            expected_bandwidth_hz
        ),
        n_timepoints=int(
            target_n_timepoints
        ),
    )

    print()
    print("[Resources] Frequency-domain pool created:")
    print(
        f"  subjects: {pool.n_subjects}"
    )
    print(
        "  water_spectra: "
        f"{tuple(pool.water_spectra.shape)}"
    )
    print(
        "  lipid_spectra: "
        f"{tuple(pool.lipid_spectra.shape)}"
    )
    print(
        "  water_offsets: "
        f"{tuple(pool.water_offsets.shape)}"
    )
    print(
        "  lipid_offsets: "
        f"{tuple(pool.lipid_offsets.shape)}"
    )

    if (
        pool.lipid_projection_operators
        is None
    ):
        print(
            "  projection operators: not loaded"
        )
    else:
        print(
            "  projection operators: "
            f"{tuple(pool.lipid_projection_operators.shape)}"
        )

    return pool


def build_simulation_resources(
    *,
    train_cfg: TrainConfig,
    simulation_cfg: SimulationConfig,
) -> SimulationResources:
    """
    Build train and validation simulation pools from the complete
    training and simulation configurations.

    The HDF5 resources are stored as FIDs. During loading they are
    transformed once, so the returned pools contain spectra.

    The returned tensors are initially stored on the CPU.
    """

    resource_filename = (
        train_cfg.data.resources.filename.format(
            version=train_cfg.data.resources.version,
        )
    )

    target_n_timepoints = (
        simulation_cfg
        .acquisition
        .n_timepoints
    )

    expected_bandwidth_hz = (
        simulation_cfg
        .acquisition
        .bandwidth_hz
    )

    load_projection_operator = (
        simulation_cfg
        .lipid_projection
        .enabled
    )

    print()
    print("=" * 72)
    print("Building training simulation pool")
    print("=" * 72)

    train_pool = load_simulation_pool(
        base_dir=train_cfg.data.base_dir,
        subjects=(
            train_cfg.data.train_subjects
        ),
        resource_filename=resource_filename,
        target_n_timepoints=(
            target_n_timepoints
        ),
        expected_bandwidth_hz=(
            expected_bandwidth_hz
        ),
        load_projection_operator=(
            load_projection_operator
        ),
    )

    print()
    print("=" * 72)
    print("Building validation simulation pool")
    print("=" * 72)

    validation_pool = load_simulation_pool(
        base_dir=train_cfg.data.base_dir,
        subjects=(
            train_cfg.data.val_subjects
        ),
        resource_filename=resource_filename,
        target_n_timepoints=(
            target_n_timepoints
        ),
        expected_bandwidth_hz=(
            expected_bandwidth_hz
        ),
        load_projection_operator=(
            load_projection_operator
        ),
    )

    resources = SimulationResources(
        train=train_pool,
        validation=validation_pool,
    )

    print()
    print("=" * 72)
    print("Simulation resources ready")
    print("=" * 72)
    print(
        "Train subjects: "
        f"{resources.train.n_subjects}"
    )
    print(
        "Validation subjects: "
        f"{resources.validation.n_subjects}"
    )
    print(
        "Target spectral points: "
        f"{target_n_timepoints}"
    )
    print(
        "Bandwidth: "
        f"{expected_bandwidth_hz} Hz"
    )
    print(
        "Projection operators loaded: "
        f"{load_projection_operator}"
    )

    return resources
