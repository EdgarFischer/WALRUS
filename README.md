# WALRUS

**WALRUS (Water And Lipid Removal Using Spectral Learning)** is a deep learning framework for removing residual water and lipid contamination from proton magnetic resonance spectroscopic imaging (¹H-MRSI) data.

Building on the original WALINET approach, WALRUS introduces a fitting-consistent spectral simulator with empirically calibrated parameter distributions. The simulator combines metabolite signals generated from the same spectral basis used for downstream quantification with water and lipid signals derived from *in vivo* measurements. This enables the generation of diverse, realistic training spectra.

The final WALRUS model uses a U-Net architecture to estimate and remove water and lipid contamination directly from complex-valued MRSI spectra, without requiring subject-specific lipid masks or L2 regularization during inference.

## Features

- **Fitting-consistent spectral simulation:** Metabolite signals are generated using the spectral basis employed for downstream fitting.
- **Empirical simulator calibration:** Simulation parameter distributions are calibrated from *in vivo* spectral fitting results.
- **On-the-fly data generation:** Synthetic training spectra are generated dynamically during training.
- **Configurable neural networks:** Support for U-Net architectures with adjustable depth and channel count, as well as Y-Net architectures incorporating an auxiliary L2-derived lipid estimate.
- **Training and inference:** Configuration-driven workflows for model training and nuisance-signal removal from complex-valued MRSI data.

## Background

WALRUS builds on **WALINET**, introduced by Weiser et al. in *Magnetic Resonance in Medicine*:

- [WALINET publication](https://doi.org/10.1002/mrm.30402)
- [Original WALINET source code](https://github.com/weiserjpaul/WALINET)


## Concept

![WALRUS workflow overview](Preview.png)

## Documentation

More detailed documentation is split into:

- [Simulation](docs/simulation.md)
- [Calibration](docs/calibration.md)
- [Training](docs/training.md)
- [Inference](docs/inference.md)
- [Data formats](docs/data_formats.md)

## Repository layout

```text
WALRUS/
├── configs/                 YAML configuration files
│   ├── Simulation/          simulator and distribution configs
│   ├── Training/            model-training configs
│   └── PaperFig1/           evaluation configs for paper figures
├── docs/                    user-facing documentation
├── scripts/                 command-line workflows
│   ├── SimulationScripts/   training-data, basis, water/lipid utilities
│   ├── trainingScripts/     model training entry points
│   ├── inference/           inference scripts for prepared datasets
│   ├── DataPrepScripts/     site-specific data conversion helpers
│   └── evaluation/          paper/evaluation utilities
├── src/walinet/             Python package
│   ├── config/              config schemas and builders
│   ├── data/                data loading and CombinedCSI I/O
│   ├── inference/           FID inference and pipeline code
│   ├── model/               neural-network models
│   ├── training/            training loops and fixed-pool support
│   ├── training_data/       simulator, basis handling, resources
│   └── paper_figures/       reproducible figure helpers
└── tests/                   unit tests and optional integration tests
```

Local datasets, generated outputs, fitted maps, graphics, metabolite-mode resources, and checkpoints are intentionally kept outside version control.

## Installation

The recommended setup is a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already use an existing environment, install the same dependency set from:

```text
requirements.txt
```

During development, run commands from the repository root and ensure that `src/` is on the Python path. `pytest.ini` does this automatically for tests.

## Quick start

### 1. Prepare data

Place each subject in the expected local structure:

```text
<base_dir>/<Subject>/
├── OriginalData/
│   └── data.npy
└── masks/
    ├── brain_mask.npy
    └── lipid_mask.npy
```

See [Data formats](docs/data_formats.md) for shapes, dtypes, and naming conventions.

### 2. Prepare water and lipid simulation resources

Before on-the-fly simulation or training, extract the subject-specific water
and lipid pools and, when enabled, the lipid projection operators. Select and
edit the matching acquisition config:

```text
configs/Simulation/3T_water_lipid_v1.yaml
configs/Simulation/7T_water_lipid_v1.yaml
```

In particular, set `data.base_dir`, `data.subjects`, the input/mask paths, and
the acquisition parameters. Then run, for example:

```bash
python scripts/SimulationScripts/extract_water_lipid.py \
  configs/Simulation/7T_water_lipid_v1.yaml
```

For every configured subject this writes a resource file such as:

```text
<subject>/TrainData/SimulationResources_water_lipid_v1.h5
```

The training config must reference the same resource version and filename.
This preparation only needs to be repeated when the source data, masks,
acquisition settings, or extraction configuration change. See
[Simulation](docs/simulation.md) for details.

### 3. Configure simulation

Start from one of the existing simulator configs:

```text
configs/Simulation/3T_on_the_fly.yaml
configs/Simulation/7T_on_the_fly.yaml
configs/Simulation/7T_on_the_fly_WALINET_STYLE.yaml
```

These define acquisition parameters, simulator distributions, masks/resources, and whether acquisition length is fixed or variable.

### 4. Train

Start from:

```text
configs/Training/train_3T.yaml
configs/Training/train_7T.yaml
```

Run:

```bash
python scripts/trainingScripts/train.py --config configs/Training/train_7T.yaml
```

See [Training](docs/training.md) for warm starts, fixed training pools, and validation behavior.

### 5. Run inference

For NumPy CSI data, the core inference API is in `walinet.inference.fid_inference`. Project-specific scripts live in:

```text
scripts/inference/
```

Typical output is a corrected FID array such as:

```text
OriginalData/data_after_walrus.npy
```

See [Inference](docs/inference.md).

### 6. Test

Fast unit tests:

```bash
python -m pytest -q
```

GPU/data-heavy integration tests are marked separately and are not part of the default test run:

```bash
python -m pytest -q -m integration
```

## Notes on scope

This repository is research software. Some scripts are intentionally site-specific and encode local dataset conventions. The reusable core is the package under `src/walinet/` plus the YAML-driven simulation, training, and inference workflows.
