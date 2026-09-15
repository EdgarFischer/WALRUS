from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from walinet.paper_figures.mask_degradation.config import ModelConfig


@dataclass(frozen=True)
class TraceSweepConfig:
    source_path: Path
    training_config: Path
    models: tuple[ModelConfig, ...]
    subjects: tuple[str, ...]
    n_spectra_per_subject: int
    batch_size: int
    seed: int
    device: str
    targets: tuple[float, ...]
    input_data: str
    lipid_mask: str
    output_path: Path


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path) -> TraceSweepConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("Trace-sweep config must be a YAML mapping.")

    raw_models = raw["models"]
    models = tuple(
        ModelConfig(
            label=str(item["label"]),
            path=_resolve(item["path"], path.parent),
            checkpoint=str(item.get("checkpoint", "model_best.pt")),
        )
        for item in raw_models
    )
    subjects = tuple(str(value) for value in raw["simulation"]["subjects"])
    targets = tuple(float(value) for value in raw["trace_sweep"]["targets"])
    if not models or len({item.label for item in models}) != len(models):
        raise ValueError("models must be non-empty and have unique labels.")
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("simulation.subjects must be non-empty and unique.")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("trace_sweep.targets must be non-empty and unique.")
    if any(not 0.0 < target <= 1.0 for target in targets):
        raise ValueError("Every trace target must be in (0, 1].")

    simulation = raw["simulation"]
    n_spectra = int(simulation.get("n_spectra_per_subject", 10_000))
    batch_size = int(simulation.get("batch_size", 1_000))
    if n_spectra <= 0 or batch_size <= 0:
        raise ValueError("Spectrum count and batch size must be positive.")
    data = raw.get("data", {})
    return TraceSweepConfig(
        source_path=path,
        training_config=_resolve(raw["training_config"], path.parent),
        models=models,
        subjects=subjects,
        n_spectra_per_subject=n_spectra,
        batch_size=batch_size,
        seed=int(simulation.get("seed", 42)),
        device=str(raw.get("device", "cuda:0")),
        targets=targets,
        input_data=str(data.get("input_data", "OriginalData/data.npy")),
        lipid_mask=str(data.get("lipid_mask", "masks/lipid_mask.npy")),
        output_path=_resolve(raw["output"]["path"], path.parent),
    )
