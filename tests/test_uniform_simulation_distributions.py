from pathlib import Path

import torch
import yaml

from walinet.config.build_simulation import build_simulation_config
from walinet.config.schema_simulation import UniformDistributionCfg
from walinet.training_data.distributions import sample_distribution


def test_uniform_distribution_sampling_is_seeded_and_bounded() -> None:
    distribution = UniformDistributionCfg(minimum=-0.5, maximum=0.5)
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first = sample_distribution(
        distribution=distribution,
        shape=(10_000,),
        device="cpu",
        dtype=torch.float64,
        generator=first_generator,
    )
    second = sample_distribution(
        distribution=distribution,
        shape=(10_000,),
        device="cpu",
        dtype=torch.float64,
        generator=second_generator,
    )

    assert torch.equal(first, second)
    assert bool(torch.all(first >= -0.5))
    assert bool(torch.all(first < 0.5))
    assert abs(float(first.mean())) < 0.02


def test_walinet_style_uniform_ranges_and_unit_conversion() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / "configs/Simulation/7T_on_the_fly_WALINET_STYLE.yaml"
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = build_simulation_config(raw, config_dir=config_path.parent)
    hz_per_ppm = config.acquisition.nmr_frequency_hz / 1e6

    phase = config.metabolites.zero_order_phase.distribution
    shift = config.metabolites.frequency_shift.distribution
    fwhm = config.metabolites.fwhm.distribution

    assert isinstance(phase, UniformDistributionCfg)
    assert phase.minimum == 0.0
    assert phase.maximum == 2.0 * torch.pi

    assert isinstance(shift, UniformDistributionCfg)
    assert shift.minimum == -0.5 * hz_per_ppm
    assert shift.maximum == 0.5 * hz_per_ppm

    assert isinstance(fwhm, UniformDistributionCfg)
    assert fwhm.minimum == 0.0135 * hz_per_ppm
    assert fwhm.maximum == 0.168 * hz_per_ppm
