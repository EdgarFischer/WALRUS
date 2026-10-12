from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time

import torch


SUPPORTED_ARCHITECTURES = {
    "unet",
    "ynet",
}


@dataclass(frozen=True)
class FixedNetworkBatch:
    """
    One fixed validation batch.

    The validation set is generated exactly once before training.
    Only the final network tensors are retained, normally on the
    CPU.

    Shapes
    ------
    network_input:
        (B, 2, T)

    network_target:
        (B, 2, T)

    network_l2:
        (B, 2, T), or None
    """

    network_input: torch.Tensor
    network_target: torch.Tensor
    network_l2: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(
            self.network_input.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.network_input.shape[-1]
        )


def normalize_architecture(
    architecture: str,
) -> str:
    """
    Validate and normalize an architecture name.
    """
    architecture = str(
        architecture
    ).lower()

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Use 'ynet' or 'unet'."
        )

    return architecture


def validate_network_tensor(
    *,
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Validate one real/imaginary network tensor.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape (B, C, T), "
            f"but found {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != 2:
        raise ValueError(
            f"{name} must contain two channels "
            "(real and imaginary), but found "
            f"{tensor.shape[1]} channels."
        )

    if torch.is_complex(
        tensor
    ):
        raise TypeError(
            f"{name} must be real-valued."
        )

    if not torch.is_floating_point(
        tensor
    ):
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )


def validate_fixed_batch(
    *,
    batch: FixedNetworkBatch,
    architecture: str,
) -> None:
    """
    Validate one stored validation batch.
    """
    architecture = normalize_architecture(
        architecture
    )

    validate_network_tensor(
        tensor=batch.network_input,
        name="network_input",
    )

    validate_network_tensor(
        tensor=batch.network_target,
        name="network_target",
    )

    if (
        batch.network_input.shape
        != batch.network_target.shape
    ):
        raise ValueError(
            "Validation input and target must have identical "
            "shapes:\n"
            f"  input:  "
            f"{tuple(batch.network_input.shape)}\n"
            f"  target: "
            f"{tuple(batch.network_target.shape)}"
        )

    if batch.network_l2 is not None:
        validate_network_tensor(
            tensor=batch.network_l2,
            name="network_l2",
        )

        if (
            batch.network_l2.shape
            != batch.network_input.shape
        ):
            raise ValueError(
                "Validation L2 input and primary input must "
                "have identical shapes:\n"
                f"  primary input: "
                f"{tuple(batch.network_input.shape)}\n"
                f"  L2 input:      "
                f"{tuple(batch.network_l2.shape)}"
            )

    if (
        architecture == "ynet"
        and batch.network_l2 is None
    ):
        raise RuntimeError(
            "YNet requires the L2-projected second input, "
            "but network_l2 is None. Enable lipid projection "
            "in the simulation configuration."
        )


def validate_prediction(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """
    Validate model output before loss calculation.
    """
    if prediction.shape != target.shape:
        raise RuntimeError(
            "Model prediction and target have different shapes:\n"
            f"  prediction: "
            f"{tuple(prediction.shape)}\n"
            f"  target:     "
            f"{tuple(target.shape)}"
        )

    if prediction.device != target.device:
        raise RuntimeError(
            "Model prediction and target are on different "
            "devices:\n"
            f"  prediction: {prediction.device}\n"
            f"  target:     {target.device}"
        )

    if not torch.is_floating_point(
        prediction
    ):
        raise TypeError(
            "Model prediction must be floating-point."
        )


def _channels_to_complex_spectra(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Convert a network tensor ``(B, 2, T)`` to complex spectra."""
    validate_network_tensor(
        tensor=tensor,
        name="network tensor",
    )
    return torch.complex(
        tensor[:, 0, :],
        tensor[:, 1, :],
    )


def _complex_spectra_to_channels(
    spectra: torch.Tensor,
) -> torch.Tensor:
    """Convert complex spectra ``(B, T)`` to a network tensor."""
    return torch.stack(
        (spectra.real, spectra.imag),
        dim=1,
    ).contiguous()


def augment_fixed_training_batch(
    *,
    network_input: torch.Tensor,
    network_target: torch.Tensor,
    network_l2: torch.Tensor | None,
    minimum_n_timepoints: int,
    maximum_n_timepoints: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply one shared random native FID length to a fixed batch.

    The fixed pool is stored at the maximum acquisition length.  A length is
    sampled for every optimizer batch, all spectra are transformed to the FID
    domain and cropped together, and the cropped FIDs are transformed back to
    spectra.  Finally, input, target, and optional L2 input are normalized with
    the cropped input maximum, matching the simulator's preparation step.
    """
    minimum_n_timepoints = int(minimum_n_timepoints)
    maximum_n_timepoints = int(maximum_n_timepoints)
    stored_n_timepoints = int(network_input.shape[-1])

    if not (0 < minimum_n_timepoints <= maximum_n_timepoints):
        raise ValueError("Invalid fixed-training acquisition-length range.")
    if stored_n_timepoints != maximum_n_timepoints:
        raise ValueError(
            "The fixed training pool must be stored at the maximum "
            "acquisition length:\n"
            f"  stored:  {stored_n_timepoints}\n"
            f"  maximum: {maximum_n_timepoints}"
        )

    if minimum_n_timepoints == maximum_n_timepoints:
        return network_input, network_target, network_l2

    crop_length = int(
        torch.randint(
            low=minimum_n_timepoints,
            high=maximum_n_timepoints + 1,
            size=(),
            generator=generator,
            device=network_input.device,
            dtype=torch.int64,
        ).item()
    )

    spectra = [
        _channels_to_complex_spectra(network_input),
        _channels_to_complex_spectra(network_target),
    ]
    if network_l2 is not None:
        spectra.append(_channels_to_complex_spectra(network_l2))

    stacked_spectra = torch.stack(spectra, dim=1)
    fids = torch.fft.ifft(
        torch.fft.ifftshift(stacked_spectra, dim=-1),
        dim=-1,
    )
    cropped_spectra = torch.fft.fftshift(
        torch.fft.fft(fids[..., :crop_length], dim=-1),
        dim=-1,
    )

    normalization_scale = torch.amax(
        torch.abs(cropped_spectra[:, 0, :]),
        dim=-1,
        keepdim=True,
    )
    if bool(torch.any(~torch.isfinite(normalization_scale))) or bool(
        torch.any(normalization_scale <= 0)
    ):
        raise RuntimeError(
            "Fixed-training crop produced an invalid normalization scale."
        )
    cropped_spectra = cropped_spectra / normalization_scale[:, None, :]

    cropped_input = _complex_spectra_to_channels(cropped_spectra[:, 0, :])
    cropped_target = _complex_spectra_to_channels(cropped_spectra[:, 1, :])
    cropped_l2 = (
        _complex_spectra_to_channels(cropped_spectra[:, 2, :])
        if network_l2 is not None
        else None
    )
    return cropped_input, cropped_target, cropped_l2


def forward_model(
    *,
    model: torch.nn.Module,
    network_input: torch.Tensor,
    network_l2: torch.Tensor | None,
    architecture: str,
) -> torch.Tensor:
    """
    Run the architecture-specific forward pass.

    UNet:
        unprojected complete input

    YNet:
        unprojected complete input
        plus L2-projected second input
    """
    architecture = normalize_architecture(
        architecture
    )

    if architecture == "unet":
        return model(
            network_input
        )

    if network_l2 is None:
        raise RuntimeError(
            "YNet requires network_l2, but no L2-projected "
            "spectrum was supplied."
        )

    return model(
        network_input,
        network_l2,
    )


@torch.no_grad()
def create_fixed_validation_batches(
    *,
    simulator,
    generator: torch.Generator,
    n_spectra: int,
    batch_size: int,
    architecture: str,
    verbose: bool = True,
) -> tuple[FixedNetworkBatch, ...]:
    """
    Generate the complete validation set exactly once.

    The simulator may run on the GPU, but only the trainer-relevant
    network tensors are retained. These tensors are copied to the
    CPU so the fixed validation set does not permanently occupy GPU
    memory.

    A separate random generator must be used for validation so that
    validation-set generation does not change the training random
    sequence.
    """
    architecture = normalize_architecture(
        architecture
    )

    n_spectra = int(
        n_spectra
    )

    batch_size = int(
        batch_size
    )

    if n_spectra <= 0:
        raise ValueError(
            "n_spectra must be > 0."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    validation_batches: list[
        FixedNetworkBatch
    ] = []

    generated_spectra = 0
    batch_index = 0

    if verbose:
        print(
            "Generating fixed validation set:"
        )
        print(
            f"  spectra:    {n_spectra}"
        )
        print(
            f"  batch size: {batch_size}"
        )
        print(
            f"  device:     {simulator.device}"
        )

    while generated_spectra < n_spectra:
        current_batch_size = min(
            batch_size,
            n_spectra
            - generated_spectra,
        )

        simulated = simulator.simulate(
            batch_size=current_batch_size,
            generator=generator,
        )

        if simulated.network_l2 is None:
            network_l2_cpu = None
        else:
            network_l2_cpu = (
                simulated
                .network_l2
                .detach()
                .cpu()
                .contiguous()
            )

        fixed_batch = FixedNetworkBatch(
            network_input=(
                simulated
                .network_input
                .detach()
                .cpu()
                .contiguous()
            ),
            network_target=(
                simulated
                .network_target
                .detach()
                .cpu()
                .contiguous()
            ),
            network_l2=(
                network_l2_cpu
            ),
        )

        validate_fixed_batch(
            batch=fixed_batch,
            architecture=architecture,
        )

        validation_batches.append(
            fixed_batch
        )

        generated_spectra += (
            current_batch_size
        )

        batch_index += 1

        if verbose:
            print(
                f"  Validation batch "
                f"{batch_index:03d}: "
                f"{generated_spectra}/"
                f"{n_spectra}"
            )

        del simulated

    if generated_spectra != n_spectra:
        raise RuntimeError(
            "Generated validation-set size does not match "
            "the requested number of spectra."
        )

    if not validation_batches:
        raise RuntimeError(
            "No validation batches were generated."
        )

    if verbose:
        print(
            "Fixed validation set ready:"
        )
        print(
            f"  batches: "
            f"{len(validation_batches)}"
        )
        print(
            f"  spectra: "
            f"{generated_spectra}"
        )

    return tuple(
        validation_batches
    )


@torch.no_grad()
def create_fixed_training_pool(
    *,
    simulator,
    generator: torch.Generator,
    n_spectra_per_subject: int,
    batch_size: int,
    architecture: str,
    verbose: bool = True,
) -> FixedNetworkBatch:
    """Generate an exactly subject-balanced training pool once."""
    architecture = normalize_architecture(architecture)
    n_spectra_per_subject = int(n_spectra_per_subject)
    batch_size = int(batch_size)
    if n_spectra_per_subject <= 0 or batch_size <= 0:
        raise ValueError(
            "n_spectra_per_subject and batch_size must be > 0."
        )

    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    l2_inputs: list[torch.Tensor] = []

    if verbose:
        print("Generating fixed training pool:")
        print(f"  subjects:           {simulator.pool.n_subjects}")
        print(f"  spectra per subject: {n_spectra_per_subject}")

    for subject_index in range(simulator.pool.n_subjects):
        generated = 0
        while generated < n_spectra_per_subject:
            current_batch_size = min(
                batch_size,
                n_spectra_per_subject - generated,
            )
            subject_indices = torch.full(
                (current_batch_size,),
                subject_index,
                dtype=torch.int64,
                device=simulator.device,
            )
            simulated = simulator.simulate(
                batch_size=current_batch_size,
                generator=generator,
                acquisition_length_override=int(
                    simulator.config.acquisition.max_acquired_n_timepoints
                ),
                subject_indices=subject_indices,
            )
            inputs.append(simulated.network_input.detach().cpu())
            targets.append(simulated.network_target.detach().cpu())
            if architecture == "ynet" and simulated.network_l2 is not None:
                l2_inputs.append(simulated.network_l2.detach().cpu())
            generated += current_batch_size

        if verbose:
            print(
                f"  Subject {subject_index + 1:03d}/"
                f"{simulator.pool.n_subjects:03d}: {generated}"
            )

    fixed_pool = FixedNetworkBatch(
        network_input=torch.cat(inputs, dim=0).contiguous(),
        network_target=torch.cat(targets, dim=0).contiguous(),
        network_l2=(
            torch.cat(l2_inputs, dim=0).contiguous()
            if l2_inputs
            else None
        ),
    )
    validate_fixed_batch(batch=fixed_pool, architecture=architecture)

    expected_size = simulator.pool.n_subjects * n_spectra_per_subject
    if fixed_pool.batch_size != expected_size:
        raise RuntimeError(
            "Fixed training-pool size does not match the requested size."
        )

    if verbose:
        print(f"Fixed training pool ready: {fixed_pool.batch_size} spectra")

    return fixed_pool


def train_one_epoch(
    *,
    model: torch.nn.Module,
    simulator,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    loss_func,
    architecture: str,
    batch_size: int,
    n_batches: int,
    verbose: bool,
    device: torch.device,
    epoch: int,
    fixed_training_pool: FixedNetworkBatch | None = None,
    fixed_generator: torch.Generator | None = None,
) -> tuple[
    torch.nn.Module,
    float,
]:
    """
    Train for one epoch from fresh simulations or a fixed pool.
    """
    architecture = normalize_architecture(
        architecture
    )

    batch_size = int(
        batch_size
    )

    n_batches = int(
        n_batches
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    if n_batches <= 0:
        raise ValueError(
            "n_batches must be > 0 for on-the-fly training."
        )

    if torch.device(
        simulator.device
    ) != torch.device(
        device
    ):
        raise RuntimeError(
            "Simulator and model must use the same device:\n"
            f"  simulator: {simulator.device}\n"
            f"  model:     {device}"
        )

    if fixed_training_pool is not None:
        validate_fixed_batch(
            batch=fixed_training_pool,
            architecture=architecture,
        )
        if fixed_generator is None:
            raise ValueError(
                "fixed_generator is required for fixed-pool training."
            )
        fixed_pool_size = fixed_training_pool.batch_size
        fixed_permutation = torch.randperm(
            fixed_pool_size,
            generator=fixed_generator,
        )
        fixed_position = 0

    model.train()

    total_weighted_loss = 0.0
    total_spectra = 0

    epoch_start = (
        time.perf_counter()
    )

    for batch_index in range(
        n_batches
    ):
        batch_start = (
            time.perf_counter()
        )

        acquisition_length_override = None

        if (
            epoch == 0
            and batch_index == 0
            and not simulator.config.acquisition.zero_filling
        ):
            acquisition_length_override = int(
                simulator
                .config
                .acquisition
                .max_acquired_n_timepoints
            )

        if fixed_training_pool is None:
            simulated = simulator.simulate(
                batch_size=batch_size,
                generator=generator,
                acquisition_length_override=(
                    acquisition_length_override
                ),
            )
            network_input = simulated.network_input
            network_target = simulated.network_target
            network_l2 = simulated.network_l2
            retries_used = simulated.retries_used
        else:
            index_parts: list[torch.Tensor] = []
            remaining = batch_size
            while remaining:
                available = fixed_pool_size - fixed_position
                take = min(remaining, available)
                index_parts.append(
                    fixed_permutation[fixed_position:fixed_position + take]
                )
                fixed_position += take
                remaining -= take
                if fixed_position == fixed_pool_size:
                    fixed_permutation = torch.randperm(
                        fixed_pool_size,
                        generator=fixed_generator,
                    )
                    fixed_position = 0
            fixed_indices = torch.cat(index_parts)
            network_input = fixed_training_pool.network_input.index_select(
                0, fixed_indices
            ).to(device=device)
            network_target = fixed_training_pool.network_target.index_select(
                0, fixed_indices
            ).to(device=device)
            network_l2 = (
                None
                if fixed_training_pool.network_l2 is None
                else fixed_training_pool.network_l2.index_select(
                    0, fixed_indices
                ).to(device=device)
            )
            acquisition = simulator.config.acquisition
            network_input, network_target, network_l2 = (
                augment_fixed_training_batch(
                    network_input=network_input,
                    network_target=network_target,
                    network_l2=network_l2,
                    minimum_n_timepoints=int(
                        acquisition.min_acquired_n_timepoints
                    ),
                    maximum_n_timepoints=int(
                        acquisition.max_acquired_n_timepoints
                    ),
                    generator=generator,
                )
            )
            retries_used = 0

        if torch.device(network_input.device) != torch.device(device):
            raise RuntimeError(
                "Training batch is on the wrong device:\n"
                f"  batch: {network_input.device}\n"
                f"  model: {device}"
            )

        if (
            architecture == "ynet"
            and network_l2 is None
        ):
            raise RuntimeError(
                "YNet training requires an L2-projected "
                "second input, but network_l2 is None."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = forward_model(
            model=model,
            network_input=network_input,
            network_l2=network_l2,
            architecture=architecture,
        )

        validate_prediction(
            prediction=prediction,
            target=network_target,
        )

        loss = loss_func(
            prediction,
            network_target,
        )

        if loss.ndim != 0:
            raise RuntimeError(
                "The training loss function must return one "
                "scalar value."
            )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            raise FloatingPointError(
                "Training loss is non-finite at "
                f"epoch {epoch + 1}, "
                f"batch {batch_index + 1}."
            )

        loss.backward()

        optimizer.step()

        loss_value = float(
            loss.detach().item()
        )

        current_batch_size = int(network_input.shape[0])

        total_weighted_loss += (
            loss_value
            * current_batch_size
        )

        total_spectra += (
            current_batch_size
        )

        batch_time = (
            time.perf_counter()
            - batch_start
        )

        if verbose:
            log_batch = (
                " ~ Epoch: {:03d}, "
                "Batch: ({:03d}/{:03d}), "
                "Loss: {:.10f}, "
                "Time: {:.4f}, "
                "Retries: {}"
            )

            print(
                log_batch.format(
                    epoch + 1,
                    batch_index + 1,
                    n_batches,
                    loss_value,
                    batch_time,
                    retries_used,
                )
            )

        del (
            network_input,
            network_target,
            network_l2,
            prediction,
            loss,
        )
        if fixed_training_pool is None:
            del simulated

    if total_spectra <= 0:
        raise RuntimeError(
            "Training epoch did not process any spectra."
        )

    epoch_loss = (
        total_weighted_loss
        / total_spectra
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        (
            "Epoch: {:03d}, "
            "Loss: {:.10f}, "
            "Spectra: {}, "
            "Time: {:.4f}"
        ).format(
            epoch + 1,
            epoch_loss,
            total_spectra,
            epoch_time,
        )
    )

    return (
        model,
        epoch_loss,
    )


def validate_one_epoch(
    *,
    model: torch.nn.Module,
    validation_batches: Sequence[
        FixedNetworkBatch
    ],
    loss_func,
    architecture: str,
    verbose: bool,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Validate on the fixed validation set.

    No new validation data are simulated here. Every epoch is
    therefore evaluated using exactly the same spectra.
    """
    architecture = normalize_architecture(
        architecture
    )

    if len(
        validation_batches
    ) == 0:
        raise ValueError(
            "validation_batches must not be empty."
        )

    model.eval()

    total_weighted_loss = 0.0
    total_spectra = 0

    epoch_start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        for batch_index, fixed_batch in enumerate(
            validation_batches
        ):
            batch_start = (
                time.perf_counter()
            )

            validate_fixed_batch(
                batch=fixed_batch,
                architecture=architecture,
            )

            network_input = (
                fixed_batch
                .network_input
                .to(
                    device=device,
                )
            )

            network_target = (
                fixed_batch
                .network_target
                .to(
                    device=device,
                )
            )

            if fixed_batch.network_l2 is None:
                network_l2 = None
            else:
                network_l2 = (
                    fixed_batch
                    .network_l2
                    .to(
                        device=device,
                    )
                )

            prediction = forward_model(
                model=model,
                network_input=network_input,
                network_l2=network_l2,
                architecture=architecture,
            )

            validate_prediction(
                prediction=prediction,
                target=network_target,
            )

            loss = loss_func(
                prediction,
                network_target,
            )

            if loss.ndim != 0:
                raise RuntimeError(
                    "The validation loss function must return "
                    "one scalar value."
                )

            if not bool(
                torch.isfinite(
                    loss
                )
            ):
                raise FloatingPointError(
                    "Validation loss is non-finite at "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )

            loss_value = float(
                loss.item()
            )

            current_batch_size = (
                fixed_batch.batch_size
            )

            total_weighted_loss += (
                loss_value
                * current_batch_size
            )

            total_spectra += (
                current_batch_size
            )

            batch_time = (
                time.perf_counter()
                - batch_start
            )

            if verbose:
                print(
                    (
                        " ~ ValEp: {:03d}, "
                        "Batch: ({:03d}/{:03d}), "
                        "Loss: {:.10f}, "
                        "Time: {:.4f}"
                    ).format(
                        epoch + 1,
                        batch_index + 1,
                        len(
                            validation_batches
                        ),
                        loss_value,
                        batch_time,
                    )
                )

            del (
                network_input,
                network_target,
                network_l2,
                prediction,
                loss,
            )

    if total_spectra <= 0:
        raise RuntimeError(
            "Validation did not process any spectra."
        )

    epoch_loss = (
        total_weighted_loss
        / total_spectra
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        (
            "ValEp: {:03d}, "
            "Loss: {:.10f}, "
            "Spectra: {}, "
            "Time: {:.4f}"
        ).format(
            epoch + 1,
            epoch_loss,
            total_spectra,
            epoch_time,
        )
    )

    return epoch_loss
