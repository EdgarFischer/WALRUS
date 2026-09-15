from pathlib import Path

import numpy as np

from walinet.paper_figures.mask_degradation.config import load_config
from walinet.paper_figures.mask_degradation.masks import dilate_lipid_inward


def test_zero_dilation_preserves_masks() -> None:
    brain = np.zeros((5, 5, 5), dtype=bool)
    brain[1:4, 1:4, 1:4] = True
    lipid = np.zeros_like(brain)
    lipid[0, :, :] = True

    brain_out, lipid_out, transferred = dilate_lipid_inward(brain, lipid, 0)

    np.testing.assert_array_equal(brain_out, brain)
    np.testing.assert_array_equal(lipid_out, lipid)
    assert not np.any(transferred)


def test_dilation_moves_only_adjacent_brain_voxels_to_lipid() -> None:
    brain = np.zeros((5, 3, 3), dtype=bool)
    brain[1:4, 1, 1] = True
    lipid = np.zeros_like(brain)
    lipid[0, 1, 1] = True

    brain_out, lipid_out, transferred = dilate_lipid_inward(brain, lipid, 2)

    assert transferred.sum() == 2
    assert lipid_out[1, 1, 1]
    assert lipid_out[2, 1, 1]
    assert brain_out[3, 1, 1]
    assert not np.any(brain_out & lipid_out)


def test_negative_dilation_moves_only_adjacent_lipid_voxels_to_brain() -> None:
    brain = np.zeros((5, 3, 3), dtype=bool)
    brain[0, 1, 1] = True
    lipid = np.zeros_like(brain)
    lipid[1:5, 1, 1] = True

    brain_out, lipid_out, transferred = dilate_lipid_inward(brain, lipid, -2)

    assert transferred.sum() == 2
    assert brain_out[1, 1, 1]
    assert brain_out[2, 1, 1]
    assert lipid_out[3, 1, 1]
    assert lipid_out[4, 1, 1]
    assert not np.any(brain_out & lipid_out)


def test_example_config_uses_requested_dilations() -> None:
    path = (
        Path(__file__).parents[1]
        / "configs/MaskDegeneration/mask_degradation.yaml"
    )
    config = load_config(path)

    assert 0 in config.dilations
    assert any(value < 0 for value in config.dilations)
    assert any(value > 0 for value in config.dilations)
    assert config.model_path.name == "YNet_ablation"
    assert config.operator_target == 0.99
