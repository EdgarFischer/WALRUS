from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


def load_subject_maps(
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    extension: str = ".nii.gz",
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load the same image map for multiple subjects and stack subjects
    along the last array dimension.

    Parameters
    ----------
    base_paths:
        Subject-specific base directories. Their order determines the
        order along the final subject dimension.

    relative_path:
        Path to the map relative to each base directory, without the
        file extension.

        Example:
            "Extra/FWHM_map"

    extension:
        File extension, for example ".nii.gz" or ".nii".

    dtype:
        NumPy dtype of the returned array.

    Returns
    -------
    np.ndarray
        Array with shape:

            (*spatial_shape, n_subjects)

        For example:

            (64, 64, 35, 9)

    Raises
    ------
    FileNotFoundError:
        If one of the requested files does not exist.

    ValueError:
        If no base paths were supplied or the image shapes differ.
    """
    base_paths = [
        Path(base_path)
        for base_path in base_paths
    ]

    if not base_paths:
        raise ValueError(
            "base_paths must contain at least one subject directory."
        )

    if not extension.startswith("."):
        extension = f".{extension}"

    relative_file = Path(
        f"{relative_path}{extension}"
    )

    subject_arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for base_path in base_paths:
        file_path = base_path / relative_file

        if not file_path.is_file():
            raise FileNotFoundError(
                "Subject map not found:\n"
                f"  {file_path}"
            )

        image = nib.load(
            str(file_path)
        )

        array = np.asarray(
            image.get_fdata(),
            dtype=dtype,
        )

        if expected_shape is None:
            expected_shape = array.shape

        elif array.shape != expected_shape:
            raise ValueError(
                "All subject maps must have the same shape:\n"
                f"  expected: {expected_shape}\n"
                f"  found:    {array.shape}\n"
                f"  file:     {file_path}"
            )

        subject_arrays.append(
            array
        )

    return np.stack(
        subject_arrays,
        axis=-1,
    )

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def load_subject_water_fids(
    *,
    base_path: str | Path,
    subject_folders: Sequence[str | Path],
    relative_path: str | Path = (
        "TrainData/"
        "SimulationResources_water_lipid_v1.h5"
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load water FIDs and brain masks for multiple subjects.

    Input per subject
    -----------------
    water_fids:
        (X, Y, Z, T)

    brain_mask:
        (X, Y, Z)

    Returns
    -------
    water_fids:
        (X, Y, Z, T, S)

    brain_mask:
        (X, Y, Z, S)
    """
    base_path = Path(base_path)
    relative_path = Path(relative_path)

    if not subject_folders:
        raise ValueError(
            "subject_folders must contain at least one subject."
        )

    water_subjects = []
    mask_subjects = []

    expected_water_shape = None
    expected_mask_shape = None

    for subject_folder in subject_folders:
        subject_folder = Path(subject_folder)

        file_path = (
            base_path
            / subject_folder
            / relative_path
        )

        if not file_path.is_file():
            raise FileNotFoundError(
                "Resource file not found:\n"
                f"{file_path}"
            )

        with h5py.File(file_path, "r") as h5:
            water_fids = np.asarray(
                h5["water_fids"][:],
                dtype=np.complex64,
            )

            brain_mask = np.asarray(
                h5["brain_mask"][:],
                dtype=bool,
            )

        if water_fids.ndim != 4:
            raise ValueError(
                f"{subject_folder}: water_fids has shape "
                f"{water_fids.shape}, expected (X, Y, Z, T)."
            )

        if brain_mask.shape != water_fids.shape[:-1]:
            raise ValueError(
                f"{subject_folder}: incompatible shapes:\n"
                f"  water: {water_fids.shape}\n"
                f"  mask:  {brain_mask.shape}"
            )

        if expected_water_shape is None:
            expected_water_shape = water_fids.shape
            expected_mask_shape = brain_mask.shape

        elif water_fids.shape != expected_water_shape:
            raise ValueError(
                "All water volumes must have the same shape:\n"
                f"  expected: {expected_water_shape}\n"
                f"  found:    {water_fids.shape}\n"
                f"  subject:  {subject_folder}"
            )

        elif brain_mask.shape != expected_mask_shape:
            raise ValueError(
                "All brain masks must have the same shape:\n"
                f"  expected: {expected_mask_shape}\n"
                f"  found:    {brain_mask.shape}\n"
                f"  subject:  {subject_folder}"
            )

        water_subjects.append(water_fids)
        mask_subjects.append(brain_mask)

    water_fids = np.stack(
        water_subjects,
        axis=-1,
    )

    brain_mask = np.stack(
        mask_subjects,
        axis=-1,
    )

    return water_fids, brain_mask


def load_subject_original_and_walinet_fids(
    *,
    base_path: str | Path,
    subject_folders: Sequence[str | Path],
    original_relative_path: str | Path = (
        "OriginalData/data.npy"
    ),
    after_walinet_relative_path: str | Path = (
        "OriginalData/data_after_walinet.npy"
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load original FIDs and FIDs after WALINET nuisance removal.

    Input per subject:
        (X, Y, Z, T)

    Returns
    -------
    original_fids:
        (X, Y, Z, T, S)

    after_walinet_fids:
        (X, Y, Z, T, S)
    """
    base_path = Path(base_path)
    original_relative_path = Path(
        original_relative_path
    )
    after_walinet_relative_path = Path(
        after_walinet_relative_path
    )

    if not subject_folders:
        raise ValueError(
            "subject_folders must contain at least one subject."
        )

    original_subjects = []
    after_walinet_subjects = []

    expected_shape = None

    for subject_folder in subject_folders:
        subject_folder = Path(subject_folder)

        original_path = (
            base_path
            / subject_folder
            / original_relative_path
        )

        after_walinet_path = (
            base_path
            / subject_folder
            / after_walinet_relative_path
        )

        if not original_path.is_file():
            raise FileNotFoundError(
                "Original FID file not found:\n"
                f"{original_path}"
            )

        if not after_walinet_path.is_file():
            raise FileNotFoundError(
                "WALINET output FID file not found:\n"
                f"{after_walinet_path}"
            )

        original_fids = np.asarray(
            np.load(
                original_path,
                allow_pickle=False,
            ),
            dtype=np.complex64,
        )

        after_walinet_fids = np.asarray(
            np.load(
                after_walinet_path,
                allow_pickle=False,
            ),
            dtype=np.complex64,
        )

        if original_fids.ndim != 4:
            raise ValueError(
                f"{subject_folder}: data.npy has shape "
                f"{original_fids.shape}, expected (X, Y, Z, T)."
            )

        if after_walinet_fids.shape != original_fids.shape:
            raise ValueError(
                f"{subject_folder}: incompatible shapes:\n"
                f"  original:      {original_fids.shape}\n"
                f"  after WALINET: {after_walinet_fids.shape}"
            )

        if expected_shape is None:
            expected_shape = original_fids.shape

        elif original_fids.shape != expected_shape:
            raise ValueError(
                "All subjects must have the same shape:\n"
                f"  expected: {expected_shape}\n"
                f"  found:    {original_fids.shape}\n"
                f"  subject:  {subject_folder}"
            )

        original_subjects.append(original_fids)
        after_walinet_subjects.append(
            after_walinet_fids
        )

    original_fids = np.stack(
        original_subjects,
        axis=-1,
    )

    after_walinet_fids = np.stack(
        after_walinet_subjects,
        axis=-1,
    )

    return original_fids, after_walinet_fids


@dataclass(frozen=True)
class ScalingCalibrationSubjectData:
    """Spatially aligned inputs for water/lipid scaling calibration.

    All FID arrays have shape ``(X, Y, Z, T)``. Subjects are deliberately
    kept separate so different spatial matrix sizes can coexist.
    """

    subject_dir: Path
    full_fids: np.ndarray
    after_walinet_fids: np.ndarray
    water_fids: np.ndarray
    brain_mask: np.ndarray

    @property
    def metabolites(self) -> np.ndarray:
        """WALINET nuisance-free FIDs (metabolites plus residual noise)."""
        return self.after_walinet_fids

    @property
    def nuisance(self) -> np.ndarray:
        """Nuisance predicted by WALINET: full minus nuisance-free FIDs."""
        return self.full_fids - self.after_walinet_fids

    @property
    def lipids(self) -> np.ndarray:
        """Estimated lipid FIDs: WALINET nuisance minus isolated water."""
        return self.full_fids - self.after_walinet_fids - self.water_fids


def _load_complex_fid_volume(
    path: str | Path,
    *,
    description: str,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """Load and validate one complex spatial FID volume."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found:\n  {path}"
        )

    fids = np.load(
        path,
        allow_pickle=False,
        mmap_mode=mmap_mode,
    )

    if fids.ndim != 4:
        raise ValueError(
            f"{description} must have shape (X, Y, Z, T):\n"
            f"  file:  {path}\n"
            f"  shape: {fids.shape}"
        )

    if not np.issubdtype(fids.dtype, np.complexfloating):
        raise TypeError(
            f"{description} must be complex-valued:\n"
            f"  file:  {path}\n"
            f"  dtype: {fids.dtype}"
        )

    return fids


def load_subject_after_walinet_fids(
    subject_dir: str | Path,
    *,
    relative_path: str | Path = "OriginalData/data_after_walinet.npy",
    mmap_mode: str | None = "r",
) -> np.ndarray:
    """Load one subject's spatially resolved nuisance-free WALINET FIDs."""
    subject_dir = Path(subject_dir).expanduser().resolve()

    return _load_complex_fid_volume(
        subject_dir / relative_path,
        description="WALINET nuisance-free FID volume",
        mmap_mode=mmap_mode,
    )


def load_subject_simulation_water_fids(
    subject_dir: str | Path,
    *,
    resource_relative_path: str | Path = (
        "TrainData/SimulationResources_water_lipid_v1.h5"
    ),
    verify_external_mask: bool = True,
    external_mask_relative_path: str | Path = "masks/brain_mask.npy",
) -> tuple[np.ndarray, np.ndarray]:
    """Load spatial water FIDs and their brain mask from simulation resources.

    The HDF5 brain mask is authoritative because it is stored together with
    ``water_fids``. When requested and available, the external ``.npy`` mask
    is checked for exact agreement but is not used as a second data source.
    """
    subject_dir = Path(subject_dir).expanduser().resolve()
    resource_path = subject_dir / resource_relative_path

    if not resource_path.is_file():
        raise FileNotFoundError(
            "Simulation resource file not found:\n"
            f"  {resource_path}"
        )

    with h5py.File(resource_path, "r") as h5:
        for key in ("water_fids", "brain_mask"):
            if key not in h5:
                raise KeyError(
                    f"Dataset {key!r} is missing from:\n  {resource_path}"
                )

        water_fids = np.asarray(
            h5["water_fids"][:],
            dtype=np.complex64,
        )
        brain_mask = np.asarray(
            h5["brain_mask"][:],
            dtype=bool,
        )

        domain = h5.attrs.get("domain")
        if isinstance(domain, bytes):
            domain = domain.decode("utf-8")
        if domain is not None and str(domain).lower() != "fid":
            raise ValueError(
                "Simulation water must be stored in the FID domain:\n"
                f"  file:   {resource_path}\n"
                f"  domain: {domain}"
            )

    if water_fids.ndim != 4:
        raise ValueError(
            "water_fids must have shape (X, Y, Z, T):\n"
            f"  file:  {resource_path}\n"
            f"  shape: {water_fids.shape}"
        )

    if brain_mask.shape != water_fids.shape[:-1]:
        raise ValueError(
            "HDF5 water and brain mask have incompatible shapes:\n"
            f"  file:  {resource_path}\n"
            f"  water: {water_fids.shape}\n"
            f"  mask:  {brain_mask.shape}"
        )

    if not np.all(np.isfinite(water_fids)):
        raise ValueError(
            f"water_fids contains NaN or Inf:\n  {resource_path}"
        )

    if verify_external_mask:
        external_mask_path = subject_dir / external_mask_relative_path
        if external_mask_path.is_file():
            external_mask = np.asarray(
                np.load(external_mask_path, allow_pickle=False),
                dtype=bool,
            )
            if external_mask.shape != brain_mask.shape:
                raise ValueError(
                    "External and HDF5 brain masks have different shapes:\n"
                    f"  external: {external_mask.shape}\n"
                    f"  HDF5:     {brain_mask.shape}\n"
                    f"  subject:  {subject_dir}"
                )
            if not np.array_equal(external_mask, brain_mask):
                raise ValueError(
                    "External and HDF5 brain masks are not identical:\n"
                    f"  subject: {subject_dir}"
                )

    return water_fids, brain_mask


def load_subject_scaling_calibration_data(
    subject_dir: str | Path,
    *,
    original_relative_path: str | Path = "OriginalData/data.npy",
    after_walinet_relative_path: str | Path = (
        "OriginalData/data_after_walinet.npy"
    ),
    resource_relative_path: str | Path = (
        "TrainData/SimulationResources_water_lipid_v1.h5"
    ),
    mmap_mode: str | None = "r",
    verify_external_mask: bool = True,
) -> ScalingCalibrationSubjectData:
    """Load and validate all aligned inputs for one scaling-calibration subject."""
    subject_dir = Path(subject_dir).expanduser().resolve()

    full_fids = _load_complex_fid_volume(
        subject_dir / original_relative_path,
        description="Original full FID volume",
        mmap_mode=mmap_mode,
    )
    after_walinet_fids = load_subject_after_walinet_fids(
        subject_dir,
        relative_path=after_walinet_relative_path,
        mmap_mode=mmap_mode,
    )
    water_fids, brain_mask = load_subject_simulation_water_fids(
        subject_dir,
        resource_relative_path=resource_relative_path,
        verify_external_mask=verify_external_mask,
    )

    if after_walinet_fids.shape != full_fids.shape:
        raise ValueError(
            "Original and WALINET FID volumes have different shapes:\n"
            f"  subject:        {subject_dir}\n"
            f"  original:       {full_fids.shape}\n"
            f"  after WALINET:  {after_walinet_fids.shape}"
        )

    if water_fids.shape != full_fids.shape:
        raise ValueError(
            "Original and water FID volumes have different shapes:\n"
            f"  subject:  {subject_dir}\n"
            f"  original: {full_fids.shape}\n"
            f"  water:    {water_fids.shape}"
        )

    if brain_mask.shape != full_fids.shape[:-1]:
        raise ValueError(
            "Brain mask does not match the FID spatial shape:\n"
            f"  subject: {subject_dir}\n"
            f"  FIDs:    {full_fids.shape}\n"
            f"  mask:    {brain_mask.shape}"
        )

    return ScalingCalibrationSubjectData(
        subject_dir=subject_dir,
        full_fids=full_fids,
        after_walinet_fids=after_walinet_fids,
        water_fids=water_fids,
        brain_mask=brain_mask,
    )


def load_scaling_calibration_subjects(
    subject_dirs: Sequence[str | Path],
    **loader_kwargs,
) -> list[ScalingCalibrationSubjectData]:
    """Load subjects independently without stacking their spatial dimensions."""
    subject_dirs = list(subject_dirs)

    if not subject_dirs:
        raise ValueError("subject_dirs must contain at least one subject.")

    subjects = [
        load_subject_scaling_calibration_data(
            subject_dir,
            **loader_kwargs,
        )
        for subject_dir in subject_dirs
    ]

    for subject in subjects:
        print(
            f"Loaded {subject.subject_dir}: "
            f"FIDs {subject.full_fids.shape}, "
            f"brain voxels {int(subject.brain_mask.sum())}"
        )

    return subjects

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def load_subject_reconstructed_water_fids(
    *,
    base_path: str | Path,
    subject_folders: Sequence[str | Path],
    original_relative_path: str | Path = "OriginalData/data.npy",
    suppressed_relative_path: str | Path = (
        "OriginalData/SupressedWater.npy"
    ),
    mask_relative_path: str | Path = "masks/brain_mask.npy",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct and stack water FIDs:

        Water = FullData - SupressedWater

    Returns
    -------
    water_fids:
        Shape (X, Y, Z, T, S)

    brain_mask:
        Shape (X, Y, Z, S)
    """
    base_path = Path(base_path)

    if not subject_folders:
        raise ValueError(
            "subject_folders must contain at least one subject."
        )

    water_subjects = []
    mask_subjects = []
    expected_shape = None

    for subject_folder in subject_folders:
        subject_path = base_path / Path(subject_folder)

        full_path = subject_path / original_relative_path
        suppressed_path = subject_path / suppressed_relative_path
        mask_path = subject_path / mask_relative_path

        for path in (full_path, suppressed_path, mask_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"File not found:\n{path}"
                )

        full_data = np.asarray(
            np.load(full_path, allow_pickle=False),
            dtype=np.complex64,
        )

        suppressed_water = np.asarray(
            np.load(suppressed_path, allow_pickle=False),
            dtype=np.complex64,
        )

        brain_mask = np.asarray(
            np.load(mask_path, allow_pickle=False),
            dtype=bool,
        )

        if full_data.shape != suppressed_water.shape:
            raise ValueError(
                f"{subject_folder}: incompatible FID shapes:\n"
                f"  full:       {full_data.shape}\n"
                f"  suppressed: {suppressed_water.shape}"
            )

        if brain_mask.shape != full_data.shape[:-1]:
            raise ValueError(
                f"{subject_folder}: incompatible mask shape:\n"
                f"  FIDs: {full_data.shape}\n"
                f"  mask: {brain_mask.shape}"
            )

        if expected_shape is None:
            expected_shape = full_data.shape
        elif full_data.shape != expected_shape:
            raise ValueError(
                "All subjects must have the same FID shape."
            )

        water_fids = full_data - suppressed_water

        water_subjects.append(water_fids)
        mask_subjects.append(brain_mask)

    Water = np.stack(
        water_subjects,
        axis=-1,
    )

    brain_mask = np.stack(
        mask_subjects,
        axis=-1,
    )

    return Water, brain_mask

from pathlib import Path
import re
import nibabel as nib
import numpy as np


def _metab_key(filename, suffix):
    name = filename.removesuffix(f"_amp_map{suffix}")
    name = re.sub(r"^\d+[_-]+", "", name)
    return re.sub(r"[^a-zA-Z0-9]", "", name)


def load_calibration_maps(subject_dirs, suffix=".nii.gz"):
    data = {}

    for subject_dir in map(Path, subject_dirs):
        if subject_dir.name.startswith("LCModel"):
            subject_id = (
                f"{subject_dir.parent.parent.name}/"
                f"{subject_dir.parent.name}"
            )
        else:
            subject_id = f"{subject_dir.parent.name}/{subject_dir.name}"
        orig_dir = subject_dir / "maps" / "Orig"

        metabolite_paths = sorted(
            orig_dir.glob(f"*_amp_map{suffix}")
        )

        metabolites = {
            _metab_key(path.name, suffix): nib.load(path).get_fdata(
                dtype=np.float32
            )
            for path in metabolite_paths
            if not _metab_key(path.name, suffix).lower().startswith("mm")
            and "macromol" not in _metab_key(path.name, suffix).lower()
        }

        fwhm_path = (
            subject_dir
            / "maps"
            / "Extra"
            / f"FWHM_map{suffix}"
        )

        if not metabolites:
            raise FileNotFoundError(
                f"Keine Metabolitenkarten gefunden: {orig_dir}"
            )

        if not fwhm_path.exists():
            raise FileNotFoundError(
                f"FWHM-Map nicht gefunden: {fwhm_path}"
            )

        fwhm = nib.load(fwhm_path).get_fdata(
            dtype=np.float32
        )

        data[subject_id] = {
            "metabolites": metabolites,
            "fwhm": fwhm,
            "subject_dir": subject_dir,
        }

    return data


from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np


def load_subject_maps_by_subject(
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    extension: str = ".nii.gz",
    *,
    dtype=np.float32,
) -> dict[str, np.ndarray]:

    if not extension.startswith("."):
        extension = f".{extension}"

    relative_file = Path(
        f"{relative_path}{extension}"
    )

    subject_maps = {}

    for base_path in map(Path, base_paths):
        file_path = base_path / relative_file

        if not file_path.is_file():
            raise FileNotFoundError(
                f"Subject map not found:\n  {file_path}"
            )

        subject_maps[str(base_path)] = np.asarray(
            nib.load(str(file_path)).get_fdata(),
            dtype=dtype,
        )

    return subject_maps
