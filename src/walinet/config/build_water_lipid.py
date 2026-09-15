# src/walinet/config/build_water_lipid.py

from __future__ import annotations

import math
from pathlib import Path

from .schema_water_lipid import (
    LipidProjectionCfg,
    WaterExtractionCfg,
    WaterLipidDataCfg,
    WaterLipidDataPathsCfg,
    WaterLipidExtractionConfig,
    WaterLipidResourcesCfg,
)


def _calculate_hz_per_ppm(
    nmr_frequency_hz: float,
) -> float:
    """
    Calculate the frequency difference in Hz corresponding to 1 ppm.

    Example
    -------
    At approximately 7 T proton frequency:

        297222931 Hz / 1e6
        = 297.222931 Hz/ppm
    """
    nmr_frequency_hz = float(
        nmr_frequency_hz
    )

    if (
        not math.isfinite(nmr_frequency_hz)
        or nmr_frequency_hz <= 0
    ):
        raise ValueError(
            "acquisition.nmr_frequency_hz must be finite "
            "and greater than zero."
        )

    return (
        nmr_frequency_hz
        / 1e6
    )


def _ppm_to_hz(
    value_ppm: float,
    *,
    hz_per_ppm: float,
) -> float:
    """
    Convert a frequency offset from ppm to Hz.

    The sign is preserved:

        positive ppm offset -> positive Hz offset
        negative ppm offset -> negative Hz offset
    """
    value_ppm = float(
        value_ppm
    )

    if not math.isfinite(
        value_ppm
    ):
        raise ValueError(
            "Water-extraction frequency offsets in ppm "
            "must be finite."
        )

    return (
        value_ppm
        * hz_per_ppm
    )


def validate_water_lipid_extraction_config(
    cfg: WaterLipidExtractionConfig,
) -> None:
    """
    Validate a water/lipid preprocessing configuration.

    The typed runtime configuration continues to use Hz internally:

        water_extraction.bandwidth
        water_extraction.min_freq
        water_extraction.max_freq
    """

    # ---------------------------------------------------------
    # General
    # ---------------------------------------------------------
    if not cfg.version.strip():
        raise ValueError(
            "version must not be empty."
        )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    if not cfg.data.base_dir.strip():
        raise ValueError(
            "data.base_dir must not be empty."
        )

    if not cfg.data.subjects:
        raise ValueError(
            "data.subjects must not be empty."
        )

    empty_subject_indices = [
        index
        for index, subject in enumerate(
            cfg.data.subjects
        )
        if not subject.strip()
    ]

    if empty_subject_indices:
        raise ValueError(
            "data.subjects contains empty entries at "
            f"indices {empty_subject_indices}."
        )

    if len(cfg.data.subjects) != len(
        set(cfg.data.subjects)
    ):
        raise ValueError(
            "data.subjects contains duplicate entries."
        )

    path_values = {
        "brain_mask": cfg.data.paths.brain_mask,
        "lipid_mask": cfg.data.paths.lipid_mask,
        "input_data": cfg.data.paths.input_data,
        "output_dir": cfg.data.paths.output_dir,
    }

    for name, value in path_values.items():
        if not value.strip():
            raise ValueError(
                f"data.paths.{name} must not be empty."
            )

    # ---------------------------------------------------------
    # Water extraction
    # ---------------------------------------------------------
    bandwidth_hz = float(
        cfg.water_extraction.bandwidth
    )

    min_freq_hz = float(
        cfg.water_extraction.min_freq
    )

    max_freq_hz = float(
        cfg.water_extraction.max_freq
    )

    if (
        not math.isfinite(bandwidth_hz)
        or bandwidth_hz <= 0
    ):
        raise ValueError(
            "water_extraction.bandwidth must be finite "
            "and > 0."
        )

    if cfg.water_extraction.hsvd_components <= 0:
        raise ValueError(
            "water_extraction.hsvd_components must be > 0."
        )

    if (
        not math.isfinite(min_freq_hz)
        or not math.isfinite(max_freq_hz)
    ):
        raise ValueError(
            "water_extraction.min_freq and "
            "water_extraction.max_freq must be finite."
        )

    if min_freq_hz >= max_freq_hz:
        raise ValueError(
            "water_extraction.min_freq must be smaller than "
            "water_extraction.max_freq."
        )

    nyquist_hz = (
        bandwidth_hz
        / 2.0
    )

    if (
        min_freq_hz < -nyquist_hz
        or max_freq_hz > nyquist_hz
    ):
        raise ValueError(
            "The water-extraction frequency interval exceeds "
            "the acquisition frequency range:\n"
            f"  allowed: [{-nyquist_hz}, {nyquist_hz}] Hz\n"
            f"  found:   [{min_freq_hz}, {max_freq_hz}] Hz"
        )

    if cfg.water_extraction.parallel_jobs <= 0:
        raise ValueError(
            "water_extraction.parallel_jobs must be > 0."
        )

    if cfg.water_extraction.slice_batch_size <= 0:
        raise ValueError(
            "water_extraction.slice_batch_size must be > 0."
        )

    # ---------------------------------------------------------
    # Resources
    # ---------------------------------------------------------
    resources_filename = (
        cfg.resources.simulation_resources_filename
    )

    if not resources_filename.strip():
        raise ValueError(
            "resources.simulation_resources_filename "
            "must not be empty."
        )

    if "{version}" not in resources_filename:
        raise ValueError(
            "resources.simulation_resources_filename "
            "must contain '{version}'."
        )

    # ---------------------------------------------------------
    # Lipid projection
    # ---------------------------------------------------------
    if cfg.lipid_projection.enabled:
        if not cfg.lipid_projection.n_timepoints:
            raise ValueError(
                "lipid_projection.n_timepoints must not be "
                "empty when lipid projection is enabled."
            )

        if any(
            n_timepoints <= 0
            for n_timepoints
            in cfg.lipid_projection.n_timepoints
        ):
            raise ValueError(
                "All lipid_projection.n_timepoints values "
                "must be > 0."
            )

        if len(
            cfg.lipid_projection.n_timepoints
        ) != len(
            set(cfg.lipid_projection.n_timepoints)
        ):
            raise ValueError(
                "lipid_projection.n_timepoints contains "
                "duplicate values."
            )

    if not (
        0 < cfg.lipid_projection.target <= 1
    ):
        raise ValueError(
            "lipid_projection.target must be in (0, 1]."
        )

    if cfg.lipid_projection.tol <= 0:
        raise ValueError(
            "lipid_projection.tol must be > 0."
        )

    if cfg.lipid_projection.max_iter <= 0:
        raise ValueError(
            "lipid_projection.max_iter must be > 0."
        )


def build_water_lipid_extraction_config(
    raw: dict,
    config_dir: Path | None = None,
) -> WaterLipidExtractionConfig:
    """
    Build a typed water/lipid extraction configuration from a
    dictionary loaded from YAML.

    The YAML uses human-readable acquisition and frequency units:

        acquisition.bandwidth_hz:
            Hz

        acquisition.nmr_frequency_hz:
            Hz

        water_extraction.min_frequency_offset_ppm:
            ppm offset relative to the internal spectral center

        water_extraction.max_frequency_offset_ppm:
            ppm offset relative to the internal spectral center

    The builder converts the ppm offsets to Hz. The existing schema
    and all downstream preprocessing code therefore remain unchanged:

        WaterExtractionCfg.bandwidth:
            Hz

        WaterExtractionCfg.min_freq:
            Hz

        WaterExtractionCfg.max_freq:
            Hz

    Relative data.base_dir paths are resolved relative to the
    directory containing the YAML configuration.
    """

    # ---------------------------------------------------------
    # Version
    # ---------------------------------------------------------
    version = str(
        raw["version"]
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    data_raw = raw["data"]
    paths_raw = data_raw["paths"]

    base_dir = Path(
        str(
            data_raw["base_dir"]
        )
    ).expanduser()

    if (
        config_dir is not None
        and not base_dir.is_absolute()
    ):
        base_dir = (
            config_dir
            / base_dir
        )

    base_dir = base_dir.resolve()

    paths = WaterLipidDataPathsCfg(
        brain_mask=str(
            paths_raw["brain_mask"]
        ),
        lipid_mask=str(
            paths_raw["lipid_mask"]
        ),
        input_data=str(
            paths_raw["input_data"]
        ),
        output_dir=str(
            paths_raw["output_dir"]
        ),
    )

    data = WaterLipidDataCfg(
        base_dir=str(
            base_dir
        ),
        subjects=[
            str(subject)
            for subject in data_raw["subjects"]
        ],
        paths=paths,
    )

    # ---------------------------------------------------------
    # Acquisition
    #
    # These values are required for the YAML-to-runtime unit
    # conversion but are not added to the existing typed schema.
    # ---------------------------------------------------------
    acquisition_raw = raw[
        "acquisition"
    ]

    bandwidth_hz = float(
        acquisition_raw[
            "bandwidth_hz"
        ]
    )

    if (
        not math.isfinite(bandwidth_hz)
        or bandwidth_hz <= 0
    ):
        raise ValueError(
            "acquisition.bandwidth_hz must be finite "
            "and greater than zero."
        )

    hz_per_ppm = _calculate_hz_per_ppm(
        acquisition_raw[
            "nmr_frequency_hz"
        ]
    )

    # ---------------------------------------------------------
    # Water extraction
    #
    # YAML:
    #     frequency offsets in ppm
    #
    # Internal WaterExtractionCfg:
    #     frequency offsets in Hz
    # ---------------------------------------------------------
    water_raw = raw[
        "water_extraction"
    ]

    min_freq_hz = _ppm_to_hz(
        water_raw[
            "min_frequency_offset_ppm"
        ],
        hz_per_ppm=hz_per_ppm,
    )

    max_freq_hz = _ppm_to_hz(
        water_raw[
            "max_frequency_offset_ppm"
        ],
        hz_per_ppm=hz_per_ppm,
    )

    water_extraction = WaterExtractionCfg(
        bandwidth=bandwidth_hz,
        hsvd_components=int(
            water_raw[
                "hsvd_components"
            ]
        ),
        min_freq=min_freq_hz,
        max_freq=max_freq_hz,
        parallel_jobs=int(
            water_raw.get(
                "parallel_jobs",
                1,
            )
        ),
        slice_batch_size=int(
            water_raw.get(
                "slice_batch_size",
                1,
            )
        ),
    )

    # ---------------------------------------------------------
    # Resources
    # ---------------------------------------------------------
    resources_raw = raw[
        "resources"
    ]

    resources = WaterLipidResourcesCfg(
        simulation_resources_filename=str(
            resources_raw[
                "simulation_resources_filename"
            ]
        ),
        overwrite=bool(
            resources_raw.get(
                "overwrite",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Lipid projection
    # ---------------------------------------------------------
    lipid_projection_raw = raw.get(
        "lipid_projection",
        {},
    )

    lipid_projection = LipidProjectionCfg(
        enabled=bool(
            lipid_projection_raw.get(
                "enabled",
                False,
            )
        ),
        n_timepoints=[
            int(n_timepoints)
            for n_timepoints
            in lipid_projection_raw.get(
                "n_timepoints",
                [],
            )
        ],
        target=float(
            lipid_projection_raw.get(
                "target",
                0.938,
            )
        ),
        tol=float(
            lipid_projection_raw.get(
                "tol",
                1e-3,
            )
        ),
        max_iter=int(
            lipid_projection_raw.get(
                "max_iter",
                60,
            )
        ),
    )

    # ---------------------------------------------------------
    # Complete internal runtime configuration
    # ---------------------------------------------------------
    cfg = WaterLipidExtractionConfig(
        version=version,
        data=data,
        water_extraction=water_extraction,
        resources=resources,
        lipid_projection=lipid_projection,
    )

    validate_water_lipid_extraction_config(
        cfg
    )

    return cfg
