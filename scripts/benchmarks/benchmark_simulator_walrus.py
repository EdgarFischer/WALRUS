#!/usr/bin/env python3
"""Benchmark on-the-fly 7T simulation with and without frozen WALRUS."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from walinet.inference.inference import (  # noqa: E402
    _load_checkpoint_state_dict,
    _load_model_and_params,
)
from walinet.training.training import forward_model  # noqa: E402
from walinet.training_data.build_simulation_system import (  # noqa: E402
    build_simulation_system,
)


def load_model(model_dir: Path, checkpoint: str, device: torch.device):
    model_class, params, architecture, resolved = _load_model_and_params(
        exp=model_dir.name, model_root=model_dir.parent, architecture="auto"
    )
    model = model_class(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    ).to(device)
    model.load_state_dict(
        _load_checkpoint_state_dict(resolved / checkpoint, device)
    )
    model.eval().requires_grad_(False)
    return model, architecture


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/Training/train_7T.yaml"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/7T_Final"
    )
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--batch-size", type=int, default=950)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    system = build_simulation_system(args.config)
    simulator = system.train_simulator
    device = system.device
    model, architecture = load_model(args.model_dir, args.checkpoint, device)
    generator = torch.Generator(device=device).manual_seed(20261113)

    def one_batch(measure: bool):
        synchronize(device)
        start = time.perf_counter()
        batch = simulator.simulate(
            batch_size=args.batch_size,
            generator=generator,
        )
        synchronize(device)
        after_simulation = time.perf_counter()
        with torch.no_grad():
            prediction = forward_model(
                model=model,
                network_input=batch.network_input,
                network_l2=batch.network_l2,
                architecture=architecture,
            )
        synchronize(device)
        after_walrus = time.perf_counter()
        if measure:
            return after_simulation - start, after_walrus - after_simulation
        del prediction, batch
        return None

    for _ in range(args.warmup):
        one_batch(False)
    simulation_times = []
    walrus_times = []
    for _ in range(args.iterations):
        simulation, walrus = one_batch(True)
        simulation_times.append(simulation)
        walrus_times.append(walrus)

    sim = statistics.mean(simulation_times)
    walrus = statistics.mean(walrus_times)
    total = sim + walrus
    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"iterations: {args.iterations}")
    print(f"simulator_seconds_per_batch: {sim:.6f}")
    print(f"walrus_seconds_per_batch: {walrus:.6f}")
    print(f"combined_seconds_per_batch: {total:.6f}")
    print(f"simulator_spectra_per_second: {args.batch_size / sim:.1f}")
    print(f"combined_spectra_per_second: {args.batch_size / total:.1f}")
    print(f"walrus_share_percent: {100.0 * walrus / total:.1f}")
    print(f"slowdown_factor: {total / sim:.2f}")
    if device.type == "cuda":
        print(
            "peak_allocated_gib: "
            f"{torch.cuda.max_memory_allocated(device) / 2**30:.3f}"
        )


if __name__ == "__main__":
    main()
