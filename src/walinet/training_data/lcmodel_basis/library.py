from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import hashlib
import json
import platform
import subprocess
import sys

import h5py
import numpy as np

from .parser import LCModelBasis

if TYPE_CHECKING:
    from .hlsvd import ProcessedLCModelBasis


BASIS_LIBRARY_FORMAT = "walinet_lcmodel_basis_library"
BASIS_LIBRARY_FORMAT_VERSION = "1.0"

DuplicatePolicy = Literal[
    "error",
    "skip",
    "replace",
]


# -------------------------------------------------------------------------
# General metadata helpers
# -------------------------------------------------------------------------
def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def compute_sha256(
    path: str | Path,
) -> str:
    """Compute the SHA-256 hash of a file."""
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_commit(
    repository_path: str | Path | None = None,
) -> str:
    """
    Return the current Git commit when available.

    Returns
    -------
    str
        Full Git commit hash, or ``"unknown"`` when unavailable.
    """
    cwd = (
        Path(repository_path)
        if repository_path is not None
        else Path.cwd()
    )

    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return "unknown"


def get_git_dirty_state(
    repository_path: str | Path | None = None,
) -> str:
    """
    Return whether the Git working tree contains uncommitted changes.

    Returns
    -------
    str
        ``"clean"``, ``"dirty"``, or ``"unknown"``.
    """
    cwd = (
        Path(repository_path)
        if repository_path is not None
        else Path.cwd()
    )

    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

        return (
            "dirty"
            if result.stdout.strip()
            else "clean"
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return "unknown"


def get_package_version(
    package_name: str,
) -> str:
    """Return an installed package version when available."""
    try:
        return version(package_name)

    except PackageNotFoundError:
        return "unknown"


def make_source_id(
    source_path: str | Path,
    source_sha256: str,
) -> str:
    """
    Generate a readable, content-addressed source identifier.
    """
    source_path = Path(source_path)

    safe_stem = "".join(
        character
        if character.isalnum()
        or character in "-_"
        else "_"
        for character in source_path.stem
    )

    return (
        f"{safe_stem}_"
        f"{source_sha256[:12]}"
    )


def make_processing_run_id() -> str:
    """Generate a unique identifier for one library-writing run."""
    return datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )


# -------------------------------------------------------------------------
# HDF5 helpers
# -------------------------------------------------------------------------
def write_array(
    group: h5py.Group,
    name: str,
    array: np.ndarray,
) -> h5py.Dataset:
    """
    Store an array in HDF5.

    Non-empty non-scalar arrays are compressed and protected with
    Fletcher32 checksums. Scalar and empty arrays are stored directly.
    """
    array = np.asarray(array)

    if name in group:
        raise ValueError(
            f"Dataset already exists: "
            f"{group.name}/{name}"
        )

    if array.ndim == 0 or array.size == 0:
        return group.create_dataset(
            name,
            data=array,
        )

    return group.create_dataset(
        name,
        data=array,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
    )


def initialize_basis_library(
    h5: h5py.File,
    *,
    processing_git_commit: str,
    git_working_tree_state: str,
) -> None:
    """Initialize metadata and groups in a new basis library."""
    h5.attrs["format"] = (
        BASIS_LIBRARY_FORMAT
    )

    h5.attrs["format_version"] = (
        BASIS_LIBRARY_FORMAT_VERSION
    )

    h5.attrs["created_utc"] = utc_now()
    h5.attrs["last_updated_utc"] = utc_now()

    h5.attrs["python_version"] = (
        sys.version
    )

    h5.attrs["platform"] = (
        platform.platform()
    )

    h5.attrs["numpy_version"] = (
        np.__version__
    )

    h5.attrs["h5py_version"] = (
        h5py.__version__
    )

    h5.attrs["hlsvdpropy_version"] = (
        get_package_version(
            "hlsvdpropy"
        )
    )

    h5.attrs["processing_git_commit"] = (
        processing_git_commit
    )

    h5.attrs["git_working_tree_state"] = (
        git_working_tree_state
    )

    h5.attrs[
        "library_description"
    ] = (
        "Native LCModel basis components, "
        "including original FIDs, HLSVD-cleaned "
        "FIDs, removed reference components, "
        "processing metadata, and embedded source "
        "BASIS files."
    )

    h5.require_group("sources")
    h5.require_group("components")
    h5.require_group("processing")


def validate_basis_library(
    h5: h5py.File,
) -> None:
    """
    Ensure that an existing HDF5 file is a compatible basis library.
    """
    stored_format = h5.attrs.get(
        "format"
    )

    if stored_format != BASIS_LIBRARY_FORMAT:
        raise ValueError(
            "The selected HDF5 file is not a "
            "compatible WALINET LCModel basis "
            "library.\n"
            f"Found format: {stored_format!r}"
        )

    stored_version = str(
        h5.attrs.get(
            "format_version"
        )
    )

    if (
        stored_version
        != BASIS_LIBRARY_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported basis-library format "
            "version.\n"
            f"Found: {stored_version}\n"
            f"Expected: "
            f"{BASIS_LIBRARY_FORMAT_VERSION}"
        )

    for required_group in (
        "sources",
        "components",
        "processing",
    ):
        if required_group not in h5:
            raise ValueError(
                f"Required HDF5 group is missing: "
                f"{required_group}"
            )


# -------------------------------------------------------------------------
# Source BASIS registration
# -------------------------------------------------------------------------
def register_source_basis(
    h5: h5py.File,
    *,
    source_basis_path: str | Path,
    basis: LCModelBasis,
) -> str:
    """
    Register and embed an original LCModel BASIS file.

    The complete original file is stored in the HDF5 library as bytes.
    Its SHA-256 hash provides a stable source-of-truth identifier.

    Returns
    -------
    str
        Stable source identifier.
    """
    source_basis_path = Path(
        source_basis_path
    ).resolve()

    if not source_basis_path.is_file():
        raise FileNotFoundError(
            "Source basis file not found: "
            f"{source_basis_path}"
        )

    source_sha256 = compute_sha256(
        source_basis_path
    )

    source_id = make_source_id(
        source_basis_path,
        source_sha256,
    )

    sources_group = h5["sources"]

    if source_id in sources_group:
        existing_group = sources_group[
            source_id
        ]

        existing_hash = str(
            existing_group.attrs[
                "sha256"
            ]
        )

        if existing_hash != source_sha256:
            raise RuntimeError(
                "Source identifier collision "
                "detected."
            )

        return source_id

    source_group = (
        sources_group.create_group(
            source_id
        )
    )

    source_group.attrs[
        "registered_utc"
    ] = utc_now()

    source_group.attrs["filename"] = (
        source_basis_path.name
    )

    source_group.attrs[
        "original_path"
    ] = str(source_basis_path)

    source_group.attrs["sha256"] = (
        source_sha256
    )

    source_group.attrs[
        "file_size_bytes"
    ] = source_basis_path.stat().st_size

    source_group.attrs["sequence"] = (
        basis.sequence
    )

    source_group.attrs["echo_time"] = (
        basis.echo_time
    )

    source_group.attrs["dwell_time"] = (
        basis.dwell_time
    )

    source_group.attrs[
        "bandwidth_hz"
    ] = basis.bandwidth

    source_group.attrs["hz_per_ppm"] = (
        basis.hz_per_ppm
    )

    source_group.attrs["n_points"] = (
        basis.n_points
    )

    source_group.attrs[
        "n_metabolites"
    ] = basis.n_metabolites

    # Embed the complete original LCModel BASIS file.
    source_bytes = np.frombuffer(
        source_basis_path.read_bytes(),
        dtype=np.uint8,
    )

    write_array(
        source_group,
        "original_file",
        source_bytes,
    )

    return source_id


# -------------------------------------------------------------------------
# Processing-run metadata
# -------------------------------------------------------------------------
def register_processing_run(
    h5: h5py.File,
    *,
    source_id: str,
    processed_basis: ProcessedLCModelBasis,
    processing_git_commit: str,
    git_working_tree_state: str,
    duplicate_policy: DuplicatePolicy,
) -> str:
    """Register one execution of the basis-processing pipeline."""
    run_id = make_processing_run_id()

    processing_group = h5[
        "processing"
    ]

    run_group = (
        processing_group.create_group(
            run_id
        )
    )

    run_group.attrs["status"] = (
        "running"
    )

    run_group.attrs["started_utc"] = (
        utc_now()
    )

    run_group.attrs["source_id"] = (
        source_id
    )

    run_group.attrs[
        "processing_git_commit"
    ] = processing_git_commit

    run_group.attrs[
        "git_working_tree_state"
    ] = git_working_tree_state

    run_group.attrs[
        "duplicate_policy"
    ] = duplicate_policy

    run_group.attrs[
        "reference_removal_method"
    ] = "HLSVDPROPY"

    run_group.attrs[
        "reference_ppm_min"
    ] = float(
        processed_basis.ppm_limits[0]
    )

    run_group.attrs[
        "reference_ppm_max"
    ] = float(
        processed_basis.ppm_limits[1]
    )

    run_group.attrs["ppm_reference"] = (
        float(
            processed_basis.ppm_reference
        )
    )

    run_group.attrs[
        "n_singular_values"
    ] = int(
        processed_basis.n_singular_values
    )

    run_group.attrs["n_fit_points"] = (
        int(
            processed_basis.n_fit_points
        )
    )

    run_group.attrs[
        "n_components_requested"
    ] = processed_basis.n_metabolites

    return run_id


# -------------------------------------------------------------------------
# HLSVD metadata
# -------------------------------------------------------------------------
def save_hlsvd_information(
    component_group: h5py.Group,
    hlsvd_info: dict,
) -> None:
    """Store the full HLSVD result for one basis component."""
    if "hlsvd" in component_group:
        del component_group["hlsvd"]

    hlsvd_group = (
        component_group.create_group(
            "hlsvd"
        )
    )

    scalar_keys = (
        "n_fit_points",
        "n_singular_values_found",
    )

    for key in scalar_keys:
        if key in hlsvd_info:
            hlsvd_group.attrs[key] = int(
                hlsvd_info[key]
            )

    if "hankel_shape" in hlsvd_info:
        hlsvd_group.attrs[
            "hankel_shape"
        ] = json.dumps(
            [
                int(value)
                for value in hlsvd_info[
                    "hankel_shape"
                ]
            ]
        )

    array_keys = (
        "singular_values",
        "frequencies_hz",
        "component_ppm",
        "damping_times",
        "amplitudes",
        "phases_deg",
        "selected",
        "selected_frequencies_hz",
        "selected_ppm",
        "selected_damping_times",
        "selected_amplitudes",
        "selected_phases_deg",
    )

    for key in array_keys:
        if key not in hlsvd_info:
            continue

        write_array(
            hlsvd_group,
            key,
            np.asarray(
                hlsvd_info[key]
            ),
        )


# -------------------------------------------------------------------------
# Individual basis components
# -------------------------------------------------------------------------
def add_basis_component(
    h5: h5py.File,
    *,
    component_name: str,
    source_id: str,
    source_component_name: str,
    source_component_index: int,
    basis: LCModelBasis,
    original_fid: np.ndarray,
    clean_fid: np.ndarray,
    removed_reference_fid: np.ndarray,
    hlsvd_info: dict,
    processed_basis: ProcessedLCModelBasis,
    processing_run_id: str,
    processing_git_commit: str,
    duplicate_policy: DuplicatePolicy = "error",
) -> None:
    """Add one native basis component to the HDF5 library."""
    if duplicate_policy not in {
        "error",
        "skip",
        "replace",
    }:
        raise ValueError(
            "Unknown duplicate policy: "
            f"{duplicate_policy}"
        )

    if "/" in component_name:
        raise ValueError(
            "Component names must not contain '/': "
            f"{component_name!r}"
        )

    if not (
        0
        <= source_component_index
        < basis.n_metabolites
    ):
        raise IndexError(
            "Invalid source component index: "
            f"{source_component_index}"
        )

    expected_source_name = (
        basis.names[
            source_component_index
        ]
    )

    if (
        expected_source_name
        != source_component_name
    ):
        raise ValueError(
            "Source component name and index "
            "do not match:\n"
            f"Index {source_component_index} "
            f"contains {expected_source_name!r}, "
            f"not {source_component_name!r}."
        )

    components_group = h5[
        "components"
    ]

    if component_name in components_group:
        if duplicate_policy == "error":
            raise ValueError(
                f"Component '{component_name}' "
                f"already exists in the library."
            )

        if duplicate_policy == "skip":
            print(
                "[Skip] Component already "
                f"exists: {component_name}"
            )
            return

        if duplicate_policy == "replace":
            del components_group[
                component_name
            ]

    original_fid = np.asarray(
        original_fid,
        dtype=np.complex128,
    )

    clean_fid = np.asarray(
        clean_fid,
        dtype=np.complex128,
    )

    removed_reference_fid = (
        np.asarray(
            removed_reference_fid,
            dtype=np.complex128,
        )
    )

    if not (
        original_fid.shape
        == clean_fid.shape
        == removed_reference_fid.shape
    ):
        raise ValueError(
            "Inconsistent FID shapes for "
            f"'{component_name}': "
            f"{original_fid.shape}, "
            f"{clean_fid.shape}, "
            f"{removed_reference_fid.shape}"
        )

    expected_shape = (
        basis.n_points,
    )

    if original_fid.shape != expected_shape:
        raise ValueError(
            f"Component '{component_name}' "
            f"has shape {original_fid.shape}; "
            f"expected {expected_shape}."
        )

    for array_name, array in (
        ("original_fid", original_fid),
        ("clean_fid", clean_fid),
        (
            "removed_reference_fid",
            removed_reference_fid,
        ),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{array_name} for "
                f"'{component_name}' contains "
                f"non-finite values."
            )

    reconstruction_error = float(
        np.max(
            np.abs(
                clean_fid
                + removed_reference_fid
                - original_fid
            )
        )
    )

    component_group = (
        components_group.create_group(
            component_name
        )
    )

    # Provenance
    component_group.attrs[
        "created_utc"
    ] = utc_now()

    component_group.attrs["source_id"] = (
        source_id
    )

    component_group.attrs[
        "source_component_name"
    ] = source_component_name

    component_group.attrs[
        "source_component_index"
    ] = int(source_component_index)

    component_group.attrs[
        "processing_run_id"
    ] = processing_run_id

    component_group.attrs[
        "processing_git_commit"
    ] = processing_git_commit

    component_group.attrs[
        "relative_scaling_preserved"
    ] = True

    component_group.attrs[
        "normalization_applied"
    ] = "none"

    component_group.attrs[
        "maximum_reconstruction_error"
    ] = reconstruction_error

    # LCModel metadata
    component_group.attrs[
        "lcmodel_id"
    ] = basis.ids[
        source_component_index
    ]

    component_group.attrs[
        "lcmodel_concentration"
    ] = float(
        basis.concentrations[
            source_component_index
        ]
    )

    component_group.attrs[
        "lcmodel_tramp"
    ] = float(
        basis.tramps[
            source_component_index
        ]
    )

    component_group.attrs[
        "lcmodel_volume"
    ] = float(
        basis.volumes[
            source_component_index
        ]
    )

    component_group.attrs[
        "lcmodel_ishift"
    ] = int(
        basis.ishifts[
            source_component_index
        ]
    )

    # Native sampling metadata
    component_group.attrs[
        "dwell_time"
    ] = basis.dwell_time

    component_group.attrs[
        "bandwidth_hz"
    ] = basis.bandwidth

    component_group.attrs[
        "hz_per_ppm"
    ] = basis.hz_per_ppm

    component_group.attrs["n_points"] = (
        original_fid.size
    )

    # Reference-removal metadata
    component_group.attrs[
        "reference_removal_method"
    ] = "HLSVDPROPY"

    component_group.attrs[
        "reference_ppm_min"
    ] = float(
        processed_basis.ppm_limits[0]
    )

    component_group.attrs[
        "reference_ppm_max"
    ] = float(
        processed_basis.ppm_limits[1]
    )

    component_group.attrs[
        "ppm_reference"
    ] = float(
        processed_basis.ppm_reference
    )

    component_group.attrs[
        "n_singular_values"
    ] = int(
        processed_basis.n_singular_values
    )

    component_group.attrs[
        "n_fit_points"
    ] = int(
        processed_basis.n_fit_points
    )

    # Native signal arrays
    write_array(
        component_group,
        "original_fid",
        original_fid.astype(
            np.complex64
        ),
    )

    write_array(
        component_group,
        "clean_fid",
        clean_fid.astype(
            np.complex64
        ),
    )

    write_array(
        component_group,
        "removed_reference_fid",
        removed_reference_fid.astype(
            np.complex64
        ),
    )

    save_hlsvd_information(
        component_group,
        hlsvd_info,
    )

    print(
        f"[Added] {component_name} "
        f"<- {source_id}:"
        f"{source_component_name}"
    )


# -------------------------------------------------------------------------
# Public build/extend function
# -------------------------------------------------------------------------
def build_or_extend_basis_library(
    output_path: str | Path,
    *,
    source_basis_path: str | Path,
    basis: LCModelBasis,
    processed_basis: ProcessedLCModelBasis,
    duplicate_policy: DuplicatePolicy = "error",
    processing_repository_path: str | Path | None = None,
) -> None:
    """
    Create or extend a native WALINET LCModel basis library.

    The original LCModel BASIS file is embedded in the HDF5 file.
    Each basis component is stored independently with its complete
    provenance, native FIDs, and HLSVD metadata.
    """
    if duplicate_policy not in {
        "error",
        "skip",
        "replace",
    }:
        raise ValueError(
            "Unknown duplicate policy: "
            f"{duplicate_policy}"
        )

    output_path = Path(output_path)

    if (
        list(processed_basis.names)
        != list(basis.names)
    ):
        raise ValueError(
            "Metabolite names or ordering "
            "differ between basis and "
            "processed_basis."
        )

    expected_shape = (
        basis.n_metabolites,
        basis.n_points,
    )

    arrays = {
        "original_fids":
            processed_basis.original_fids,

        "clean_fids":
            processed_basis.clean_fids,

        "reference_fids":
            processed_basis.reference_fids,
    }

    for name, array in arrays.items():
        array = np.asarray(array)

        if array.shape != expected_shape:
            raise ValueError(
                f"{name} has shape "
                f"{array.shape}; expected "
                f"{expected_shape}."
            )

    missing_info = [
        metabolite
        for metabolite in basis.names
        if metabolite
        not in processed_basis
        .hlsvd_info_by_metabolite
    ]

    if missing_info:
        raise ValueError(
            "Missing HLSVD metadata for: "
            + ", ".join(missing_info)
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = output_path.exists()

    processing_git_commit = (
        get_git_commit(
            processing_repository_path
        )
    )

    git_working_tree_state = (
        get_git_dirty_state(
            processing_repository_path
        )
    )

    with h5py.File(
        output_path,
        "a",
    ) as h5:
        if not file_exists:
            initialize_basis_library(
                h5,
                processing_git_commit=(
                    processing_git_commit
                ),
                git_working_tree_state=(
                    git_working_tree_state
                ),
            )

        else:
            validate_basis_library(h5)

        # Detect duplicate problems before modifying the file.
        existing_components = set(
            h5["components"].keys()
        )

        duplicate_components = (
            existing_components
            & set(basis.names)
        )

        if (
            duplicate_components
            and duplicate_policy == "error"
        ):
            duplicate_text = ", ".join(
                sorted(
                    duplicate_components
                )
            )

            raise ValueError(
                "The following components "
                "already exist in the library: "
                f"{duplicate_text}"
            )

        source_id = register_source_basis(
            h5,
            source_basis_path=(
                source_basis_path
            ),
            basis=basis,
        )

        processing_run_id = (
            register_processing_run(
                h5,
                source_id=source_id,
                processed_basis=(
                    processed_basis
                ),
                processing_git_commit=(
                    processing_git_commit
                ),
                git_working_tree_state=(
                    git_working_tree_state
                ),
                duplicate_policy=(
                    duplicate_policy
                ),
            )
        )

        processing_run_group = h5[
            "processing"
        ][processing_run_id]

        try:
            for index, metabolite in enumerate(
                basis.names
            ):
                add_basis_component(
                    h5,
                    component_name=metabolite,
                    source_id=source_id,
                    source_component_name=(
                        metabolite
                    ),
                    source_component_index=index,
                    basis=basis,
                    original_fid=(
                        processed_basis
                        .original_fids[index]
                    ),
                    clean_fid=(
                        processed_basis
                        .clean_fids[index]
                    ),
                    removed_reference_fid=(
                        processed_basis
                        .reference_fids[index]
                    ),
                    hlsvd_info=(
                        processed_basis
                        .hlsvd_info_by_metabolite[
                            metabolite
                        ]
                    ),
                    processed_basis=(
                        processed_basis
                    ),
                    processing_run_id=(
                        processing_run_id
                    ),
                    processing_git_commit=(
                        processing_git_commit
                    ),
                    duplicate_policy=(
                        duplicate_policy
                    ),
                )

            processing_run_group.attrs[
                "status"
            ] = "completed"

            processing_run_group.attrs[
                "completed_utc"
            ] = utc_now()

        except Exception:
            processing_run_group.attrs[
                "status"
            ] = "failed"

            processing_run_group.attrs[
                "failed_utc"
            ] = utc_now()

            h5.flush()
            raise

        h5.attrs["last_updated_utc"] = (
            utc_now()
        )

        h5.attrs[
            "last_processing_git_commit"
        ] = processing_git_commit

        h5.attrs[
            "last_git_working_tree_state"
        ] = git_working_tree_state

        h5.flush()

    print()
    print(
        "Basis library saved: "
        f"{output_path.resolve()}"
    )


# -------------------------------------------------------------------------
# Convenience functions for inspection and loading
# -------------------------------------------------------------------------
def inspect_basis_library(
    path: str | Path,
) -> None:
    """Print a readable overview of a basis library."""
    path = Path(path)

    with h5py.File(path, "r") as h5:
        validate_basis_library(h5)

        print("=" * 72)
        print("WALINET LCModel basis library")
        print("=" * 72)

        print("File:", path.resolve())
        print(
            "Format version:",
            h5.attrs[
                "format_version"
            ],
        )
        print(
            "Created:",
            h5.attrs[
                "created_utc"
            ],
        )
        print(
            "Last updated:",
            h5.attrs[
                "last_updated_utc"
            ],
        )

        print()
        print("Sources")
        print("-" * 72)

        for source_id, source in h5[
            "sources"
        ].items():
            print(source_id)
            print(
                "  Filename:",
                source.attrs[
                    "filename"
                ],
            )
            print(
                "  SHA-256:",
                source.attrs[
                    "sha256"
                ],
            )
            print(
                "  Sequence:",
                source.attrs[
                    "sequence"
                ],
            )
            print(
                "  Points:",
                source.attrs[
                    "n_points"
                ],
            )

        print()
        print("Components")
        print("-" * 72)

        for component_name in sorted(
            h5["components"].keys()
        ):
            component = h5[
                "components"
            ][component_name]

            print(
                f"{component_name:16s} "
                f"<- "
                f"{component.attrs['source_id']}"
            )


def load_basis_component(
    path: str | Path,
    component_name: str,
) -> dict:
    """
    Load one component including original, cleaned, and removed FIDs.
    """
    path = Path(path)

    with h5py.File(path, "r") as h5:
        validate_basis_library(h5)

        components = h5["components"]

        if component_name not in components:
            available = ", ".join(
                sorted(
                    components.keys()
                )
            )

            raise KeyError(
                f"Component not found: "
                f"{component_name!r}\n"
                f"Available components: "
                f"{available}"
            )

        component = components[
            component_name
        ]

        result = {
            "name": component_name,
            "original_fid": component[
                "original_fid"
            ][...],
            "clean_fid": component[
                "clean_fid"
            ][...],
            "removed_reference_fid":
                component[
                    "removed_reference_fid"
                ][...],
            "metadata": {
                key: component.attrs[key]
                for key in component.attrs
            },
        }

        source_id = str(
            component.attrs[
                "source_id"
            ]
        )

        source = h5[
            "sources"
        ][source_id]

        result["source"] = {
            key: source.attrs[key]
            for key in source.attrs
        }

    return result


def load_basis_fids(
    path: str | Path,
    *,
    component_names: list[str] | None = None,
    dataset_name: Literal[
        "original_fid",
        "clean_fid",
        "removed_reference_fid",
    ] = "clean_fid",
) -> tuple[
    list[str],
    np.ndarray,
    dict,
]:
    """
    Load multiple basis components as one stacked FID array.

    All selected components must have identical sampling parameters
    and array shapes.
    """
    path = Path(path)

    with h5py.File(path, "r") as h5:
        validate_basis_library(h5)

        components = h5["components"]

        if component_names is None:
            names = sorted(
                components.keys()
            )
        else:
            names = list(
                component_names
            )

        if not names:
            raise ValueError(
                "No basis components were "
                "selected."
            )

        missing = [
            name
            for name in names
            if name not in components
        ]

        if missing:
            raise KeyError(
                "Missing basis components: "
                + ", ".join(missing)
            )

        fids = [
            components[name][
                dataset_name
            ][...]
            for name in names
        ]

        first_component = components[
            names[0]
        ]

        reference_metadata = {
            "dwell_time": float(
                first_component.attrs[
                    "dwell_time"
                ]
            ),
            "bandwidth_hz": float(
                first_component.attrs[
                    "bandwidth_hz"
                ]
            ),
            "hz_per_ppm": float(
                first_component.attrs[
                    "hz_per_ppm"
                ]
            ),
            "n_points": int(
                first_component.attrs[
                    "n_points"
                ]
            ),
        }

        for name, fid in zip(
            names,
            fids,
        ):
            component = components[name]

            current_metadata = {
                "dwell_time": float(
                    component.attrs[
                        "dwell_time"
                    ]
                ),
                "bandwidth_hz": float(
                    component.attrs[
                        "bandwidth_hz"
                    ]
                ),
                "hz_per_ppm": float(
                    component.attrs[
                        "hz_per_ppm"
                    ]
                ),
                "n_points": int(
                    component.attrs[
                        "n_points"
                    ]
                ),
            }

            if fid.shape != fids[0].shape:
                raise ValueError(
                    f"Component '{name}' has "
                    f"shape {fid.shape}; expected "
                    f"{fids[0].shape}."
                )

            for key in (
                "dwell_time",
                "bandwidth_hz",
                "hz_per_ppm",
            ):
                if not np.isclose(
                    current_metadata[key],
                    reference_metadata[key],
                ):
                    raise ValueError(
                        f"Component '{name}' has "
                        f"incompatible {key}: "
                        f"{current_metadata[key]} "
                        f"instead of "
                        f"{reference_metadata[key]}."
                    )

            if (
                current_metadata["n_points"]
                != reference_metadata[
                    "n_points"
                ]
            ):
                raise ValueError(
                    f"Component '{name}' has "
                    "an incompatible number of "
                    "native points."
                )

        stacked_fids = np.stack(
            fids,
            axis=0,
        )

        source_ids = {
            name: str(
                components[name].attrs[
                    "source_id"
                ]
            )
            for name in names
        }

    metadata = {
        **reference_metadata,
        "dataset_name": dataset_name,
        "source_ids": source_ids,
    }

    return (
        names,
        stacked_fids,
        metadata,
    )
