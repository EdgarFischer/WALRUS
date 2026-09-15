from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelConfig:
    label: str
    path: Path
    checkpoint: str


@dataclass(frozen=True)
class MaskDegradationConfig:
    source_path: Path
    training_config: Path
    models: tuple[ModelConfig, ...]
    subjects: tuple[str, ...]
    n_spectra_per_subject: int
    batch_size: int
    seed: int
    device: str
    dilations: tuple[int, ...]
    operator_target: float
    input_data: str
    brain_mask: str
    lipid_mask: str
    output_path: Path

    @property
    def model_path(self) -> Path:
        """Backward-compatible access to the first configured model."""
        return self.models[0].path

    @property
    def checkpoint(self) -> str:
        """Backward-compatible access to the first configured checkpoint."""
        return self.models[0].checkpoint


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> MaskDegradationConfig:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError("Mask-degradation config must be a YAML mapping.")

    simulation = raw["simulation"]
    raw_models = raw.get("models")
    if raw_models is None:
        raw_models = [raw["model"]]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list.")
    models = tuple(
        ModelConfig(
            label=str(item.get("label", Path(item["path"]).name)),
            path=_resolve(item["path"], path.parent),
            checkpoint=str(item.get("checkpoint", "model_best.pt")),
        )
        for item in raw_models
    )
    labels = [model.label for model in models]
    if len(labels) != len(set(labels)):
        raise ValueError("Model labels must be unique.")
    degradation = raw["mask_degradation"]
    data = raw.get("data", {})
    subjects = tuple(str(item) for item in simulation["subjects"])
    dilations = tuple(int(item) for item in degradation["dilations"])
    if not subjects or len(subjects) != len(set(subjects)):
        raise ValueError("simulation.subjects must be non-empty and unique.")
    if not dilations:
        raise ValueError("mask_degradation.dilations must not be empty.")
    if len(dilations) != len(set(dilations)):
        raise ValueError("mask_degradation.dilations must be unique.")
    operator_target = float(degradation.get("operator_target", 0.938))
    if not 0.0 < operator_target <= 1.0:
        raise ValueError("mask_degradation.operator_target must be in (0, 1].")

    n_spectra = int(simulation.get("n_spectra_per_subject", 10_000))
    batch_size = int(simulation.get("batch_size", 1_000))
    if n_spectra <= 0 or batch_size <= 0:
        raise ValueError("Spectrum count and batch size must be positive.")

    return MaskDegradationConfig(
        source_path=path,
        training_config=_resolve(raw["training_config"], path.parent),
        models=models,
        subjects=subjects,
        n_spectra_per_subject=n_spectra,
        batch_size=batch_size,
        seed=int(simulation.get("seed", 42)),
        device=str(raw.get("device", "cuda:0")),
        dilations=dilations,
        operator_target=operator_target,
        input_data=str(data.get("input_data", "OriginalData/data.npy")),
        brain_mask=str(data.get("brain_mask", "masks/brain_mask.npy")),
        lipid_mask=str(data.get("lipid_mask", "masks/lipid_mask.npy")),
        output_path=_resolve(raw["output"]["path"], path.parent),
    )
