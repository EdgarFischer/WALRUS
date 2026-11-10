# src/walinet/config/build.py

from __future__ import annotations

from pathlib import Path

from .schema import (
    CheckpointCfg,
    DataCfg,
    ModelCfg,
    OptimCfg,
    OutputCfg,
    RunCfg,
    SchedulerCfg,
    SimulationResourcesCfg,
    TrainConfig,
    TrainingCfg,
    ValidationCfg,
)


VALID_MODEL_ARCHITECTURES = {
    "unet",
    "ynet",
}


def _resolve_path(
    path: str,
    config_dir: Path | None,
) -> str:
    """
    Resolve a filesystem path relative to the directory containing
    the training YAML.

    Empty paths remain empty.
    """
    path = path.strip()

    if not path:
        return ""

    resolved_path = Path(path).expanduser()

    if (
        config_dir is not None
        and not resolved_path.is_absolute()
    ):
        resolved_path = config_dir / resolved_path

    return str(resolved_path.resolve())


def _resolve_model_directory(
    model: str,
    output_base_dir: str,
) -> str:
    """
    Resolve a warm-start model directory.

    Relative model names are interpreted relative to
    output.base_dir.

    Example:

        preload_model: "ExistingModel"

    resolves to:

        <output.base_dir>/ExistingModel
    """
    model = model.strip()

    if not model:
        return ""

    model_path = Path(model).expanduser()

    if not model_path.is_absolute():
        model_path = (
            Path(output_base_dir)
            / model_path
        )

    return str(model_path.resolve())


def _validate_subjects(
    *,
    name: str,
    subjects: list[str],
) -> None:
    """
    Validate one subject list.
    """
    if not subjects:
        raise ValueError(
            f"{name} must not be empty."
        )

    empty_indices = [
        index
        for index, subject in enumerate(subjects)
        if not subject.strip()
    ]

    if empty_indices:
        raise ValueError(
            f"{name} contains empty entries at "
            f"indices {empty_indices}."
        )

    if len(subjects) != len(set(subjects)):
        raise ValueError(
            f"{name} contains duplicate entries."
        )


def validate_config(
    cfg: TrainConfig,
) -> None:
    """
    Validate the complete training configuration.
    """

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------
    if not cfg.run.name.strip():
        raise ValueError(
            "run.name must not be empty."
        )

    if cfg.run.seed < 0:
        raise ValueError(
            "run.seed must be >= 0."
        )

    if cfg.run.gpu < 0:
        raise ValueError(
            "run.gpu must be >= 0."
        )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    if not cfg.data.base_dir.strip():
        raise ValueError(
            "data.base_dir must not be empty."
        )

    _validate_subjects(
        name="data.train_subjects",
        subjects=cfg.data.train_subjects,
    )

    _validate_subjects(
        name="data.val_subjects",
        subjects=cfg.data.val_subjects,
    )

    overlapping_subjects = sorted(
        set(cfg.data.train_subjects)
        & set(cfg.data.val_subjects)
    )

    if False:
        raise ValueError(
            "Training and validation subjects must be "
            "disjoint. Overlapping subjects:\n"
            f"  {overlapping_subjects}"
        )

    if not cfg.data.simulation_config.strip():
        raise ValueError(
            "data.simulation_config must not be empty."
        )

    if not cfg.data.resources.version.strip():
        raise ValueError(
            "data.resources.version must not be empty."
        )

    resources_filename = (
        cfg.data.resources.filename
    )

    if not resources_filename.strip():
        raise ValueError(
            "data.resources.filename must not be empty."
        )

    if "{version}" not in resources_filename:
        raise ValueError(
            "data.resources.filename must contain "
            "'{version}'."
        )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    if not cfg.output.base_dir.strip():
        raise ValueError(
            "output.base_dir must not be empty."
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    if cfg.training.batch_size <= 0:
        raise ValueError(
            "training.batch_size must be > 0."
        )

    if cfg.training.epochs <= 0:
        raise ValueError(
            "training.epochs must be > 0."
        )

    if cfg.training.n_batches <= 0:
        raise ValueError(
            "training.n_batches must be > 0."
        )

    if cfg.training.mode not in {"on_the_fly", "fixed"}:
        raise ValueError(
            "training.mode must be 'on_the_fly' or 'fixed'."
        )

    if (
        cfg.training.mode == "fixed"
        and cfg.training.fixed_n_spectra_per_subject <= 0
    ):
        raise ValueError(
            "training.fixed.n_spectra_per_subject must be > 0 "
            "when training.mode is 'fixed'."
        )

    if cfg.training.fixed_seed < 0:
        raise ValueError(
            "training.fixed.seed must be >= 0."
        )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    if cfg.validation.seed < 0:
        raise ValueError(
            "validation.seed must be >= 0."
        )

    if cfg.validation.n_spectra <= 0:
        raise ValueError(
            "validation.n_spectra must be > 0."
        )

    if cfg.validation.batch_size <= 0:
        raise ValueError(
            "validation.batch_size must be > 0."
        )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    if cfg.optim.lr <= 0:
        raise ValueError(
            "optim.lr must be > 0."
        )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------
    if not cfg.scheduler.milestones:
        raise ValueError(
            "scheduler.milestones must not be empty."
        )

    if any(
        milestone <= 0
        for milestone in cfg.scheduler.milestones
    ):
        raise ValueError(
            "All scheduler.milestones must be > 0."
        )

    if (
        cfg.scheduler.milestones
        != sorted(cfg.scheduler.milestones)
    ):
        raise ValueError(
            "scheduler.milestones must be sorted "
            "in ascending order."
        )

    if (
        len(cfg.scheduler.milestones)
        != len(set(cfg.scheduler.milestones))
    ):
        raise ValueError(
            "scheduler.milestones contains "
            "duplicate entries."
        )

    if cfg.scheduler.gamma <= 0:
        raise ValueError(
            "scheduler.gamma must be > 0."
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    if (
        cfg.model.architecture
        not in VALID_MODEL_ARCHITECTURES
    ):
        raise ValueError(
            "model.architecture must be one of "
            f"{sorted(VALID_MODEL_ARCHITECTURES)}, "
            f"but found {cfg.model.architecture!r}."
        )

    if cfg.model.n_layers <= 0:
        raise ValueError(
            "model.n_layers must be > 0."
        )

    if cfg.model.n_filters <= 0:
        raise ValueError(
            "model.n_filters must be > 0."
        )

    if cfg.model.in_channels <= 0:
        raise ValueError(
            "model.in_channels must be > 0."
        )

    if cfg.model.out_channels <= 0:
        raise ValueError(
            "model.out_channels must be > 0."
        )

    if not 0 <= cfg.model.dropout < 1:
        raise ValueError(
            "model.dropout must be in [0, 1)."
        )

    # ---------------------------------------------------------
    # Checkpoint / warm start
    # ---------------------------------------------------------
    if (
        cfg.checkpoint.preload
        and not cfg.checkpoint.preload_model.strip()
    ):
        raise ValueError(
            "checkpoint.preload_model must not be empty "
            "when checkpoint.preload is true."
        )


def build_config(
    raw: dict,
    config_dir: Path | None = None,
) -> TrainConfig:
    """
    Build and validate a typed training configuration.

    Filesystem paths are resolved relative to the directory
    containing the training YAML.

    Resource filename templates remain relative because they are
    interpreted inside each subject directory.

    Relative checkpoint model names are resolved relative to
    output.base_dir.
    """

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------
    run_raw = raw["run"]

    run = RunCfg(
        name=str(
            run_raw["name"]
        ).strip(),
        seed=int(
            run_raw.get(
                "seed",
                42,
            )
        ),
        gpu=int(
            run_raw["gpu"]
        ),
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    data_raw = raw["data"]
    resources_raw = data_raw["resources"]

    data = DataCfg(
        base_dir=_resolve_path(
            str(data_raw["base_dir"]),
            config_dir,
        ),
        train_subjects=[
            str(subject).strip()
            for subject in data_raw["train_subjects"]
        ],
        val_subjects=[
            str(subject).strip()
            for subject in data_raw["val_subjects"]
        ],
        simulation_config=_resolve_path(
            str(data_raw["simulation_config"]),
            config_dir,
        ),
        resources=SimulationResourcesCfg(
            version=str(
                resources_raw["version"]
            ).strip(),
            filename=str(
                resources_raw["filename"]
            ).strip(),
        ),
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    output_raw = raw["output"]

    output = OutputCfg(
        base_dir=_resolve_path(
            str(output_raw["base_dir"]),
            config_dir,
        ),
        overwrite=bool(
            output_raw.get(
                "overwrite",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    training_raw = raw["training"]
    fixed_training_raw = training_raw.get(
        "fixed",
        {},
    )

    training = TrainingCfg(
        batch_size=int(
            training_raw["batch_size"]
        ),
        epochs=int(
            training_raw["epochs"]
        ),
        n_batches=int(
            training_raw["n_batches"]
        ),
        verbose=bool(
            training_raw.get(
                "verbose",
                False,
            )
        ),
        mode=str(
            training_raw.get(
                "mode",
                "on_the_fly",
            )
        ).strip().lower(),
        fixed_n_spectra_per_subject=int(
            fixed_training_raw.get(
                "n_spectra_per_subject",
                0,
            )
        ),
        fixed_seed=int(
            fixed_training_raw.get(
                "seed",
                24680,
            )
        ),
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    validation_raw = raw["validation"]

    validation = ValidationCfg(
        seed=int(
            validation_raw.get(
                "seed",
                12345,
            )
        ),
        n_spectra=int(
            validation_raw["n_spectra"]
        ),
        batch_size=int(
            validation_raw.get(
                "batch_size",
                training.batch_size,
            )
        ),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optim_raw = raw["optim"]

    optim = OptimCfg(
        lr=float(
            optim_raw["lr"]
        ),
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------
    scheduler_raw = raw["scheduler"]

    scheduler = SchedulerCfg(
        milestones=[
            int(milestone)
            for milestone in scheduler_raw["milestones"]
        ],
        gamma=float(
            scheduler_raw["gamma"]
        ),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model_raw = raw["model"]

    model = ModelCfg(
        architecture=str(
            model_raw.get(
                "architecture",
                "unet",
            )
        ).strip().lower(),
        n_layers=int(
            model_raw["n_layers"]
        ),
        n_filters=int(
            model_raw["n_filters"]
        ),
        in_channels=int(
            model_raw["in_channels"]
        ),
        out_channels=int(
            model_raw["out_channels"]
        ),
        dropout=float(
            model_raw.get(
                "dropout",
                0.0,
            )
        ),
    )

    # ---------------------------------------------------------
    # Checkpoint / warm start
    # ---------------------------------------------------------
    checkpoint_raw = raw.get(
        "checkpoint",
        {},
    )

    preload_model_raw = str(
        checkpoint_raw.get(
            "preload_model",
            "",
        )
    )

    checkpoint = CheckpointCfg(
        preload=bool(
            checkpoint_raw.get(
                "preload",
                False,
            )
        ),
        preload_model=_resolve_model_directory(
            model=preload_model_raw,
            output_base_dir=output.base_dir,
        ),
    )

    # ---------------------------------------------------------
    # Complete configuration
    # ---------------------------------------------------------
    cfg = TrainConfig(
        run=run,
        data=data,
        output=output,
        training=training,
        validation=validation,
        optim=optim,
        scheduler=scheduler,
        model=model,
        checkpoint=checkpoint,
    )

    validate_config(cfg)

    return cfg
