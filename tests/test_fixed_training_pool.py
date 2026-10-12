from types import SimpleNamespace

import torch

from walinet.training.training import (
    FixedNetworkBatch,
    augment_fixed_training_batch,
    create_fixed_training_pool,
    train_one_epoch,
)


class _UnusedSimulator:
    device = torch.device("cpu")
    config = SimpleNamespace(
        acquisition=SimpleNamespace(
            zero_filling=True,
            min_acquired_n_timepoints=4,
            max_acquired_n_timepoints=4,
        )
    )

    def simulate(self, **kwargs):
        raise AssertionError("Fixed training must not simulate new spectra.")


def test_fixed_training_reuses_pool_for_requested_number_of_batches() -> None:
    pool_values = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    pool_values = pool_values.expand(-1, 2, 4).contiguous()
    fixed_pool = FixedNetworkBatch(
        network_input=pool_values,
        network_target=2 * pool_values,
        network_l2=None,
    )

    seen_batch_sizes: list[int] = []
    model = torch.nn.Conv1d(2, 2, kernel_size=1)
    model.register_forward_pre_hook(
        lambda _model, args: seen_batch_sizes.append(args[0].shape[0])
    )

    train_one_epoch(
        model=model,
        simulator=_UnusedSimulator(),
        generator=torch.Generator().manual_seed(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_func=torch.nn.MSELoss(),
        architecture="unet",
        batch_size=2,
        n_batches=4,
        verbose=False,
        device=torch.device("cpu"),
        epoch=0,
        fixed_training_pool=fixed_pool,
        fixed_generator=torch.Generator().manual_seed(2),
    )

    assert seen_batch_sizes == [2, 2, 2, 2]


def test_fixed_batch_augmentation_crops_all_tensors_together() -> None:
    complex_input = torch.complex(
        torch.randn(5, 8),
        torch.randn(5, 8),
    )
    network_input = torch.stack(
        (complex_input.real, complex_input.imag),
        dim=1,
    )
    network_target = 2 * network_input
    network_l2 = 3 * network_input

    augmented = augment_fixed_training_batch(
        network_input=network_input,
        network_target=network_target,
        network_l2=network_l2,
        minimum_n_timepoints=4,
        maximum_n_timepoints=8,
        generator=torch.Generator().manual_seed(5),
    )

    augmented_input, augmented_target, augmented_l2 = augmented
    assert 4 <= augmented_input.shape[-1] <= 8
    assert augmented_target.shape == augmented_input.shape
    assert augmented_l2 is not None
    assert augmented_l2.shape == augmented_input.shape
    assert torch.allclose(augmented_target, 2 * augmented_input)
    assert torch.allclose(augmented_l2, 3 * augmented_input)
    input_complex = torch.complex(
        augmented_input[:, 0, :],
        augmented_input[:, 1, :],
    )
    assert torch.allclose(
        torch.amax(torch.abs(input_complex), dim=-1),
        torch.ones(5),
    )


class _FixedPoolRecordingSimulator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.pool = SimpleNamespace(n_subjects=2)
        self.config = SimpleNamespace(
            acquisition=SimpleNamespace(max_acquired_n_timepoints=8)
        )
        self.overrides: list[int | None] = []

    def simulate(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
        acquisition_length_override: int | None = None,
        subject_indices: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        del generator, subject_indices
        self.overrides.append(acquisition_length_override)
        values = torch.randn(batch_size, 2, acquisition_length_override)
        return SimpleNamespace(
            network_input=values,
            network_target=2 * values,
            network_l2=None,
        )


def test_fixed_pool_is_always_generated_at_maximum_length() -> None:
    simulator = _FixedPoolRecordingSimulator()
    pool = create_fixed_training_pool(
        simulator=simulator,
        generator=torch.Generator().manual_seed(1),
        n_spectra_per_subject=3,
        batch_size=2,
        architecture="unet",
        verbose=False,
    )

    assert pool.network_input.shape == (6, 2, 8)
    assert simulator.overrides == [8, 8, 8, 8]
