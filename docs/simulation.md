# Simulation

WALRUS training is driven primarily by on-the-fly simulation. Instead of storing every synthetic spectrum on disk, the simulator samples nuisance and metabolite parameters during training and produces network inputs and targets directly.

## Main configuration files

Common starting points:

```text
configs/Simulation/3T_on_the_fly.yaml
configs/Simulation/7T_on_the_fly.yaml
configs/Simulation/7T_on_the_fly_WALINET_STYLE.yaml
```

Additional configs define calibrated metabolite distributions and acquisition-specific resources:

```text
configs/Simulation/Metabos_healthy_3T_2sigma.yaml
configs/Simulation/Metabos_healthy_7T_2sigma.yaml
configs/Simulation/Metabos_healthy_7T_WALINET_Style.yaml
configs/Simulation/3T_water_lipid_v1.yaml
configs/Simulation/7T_water_lipid_v1.yaml
```

## Simulator concept

The simulator combines:

1. metabolite basis spectra;
2. sampled metabolite amplitudes;
3. frequency shift, phase, linewidth, and SNR distributions;
4. nuisance water and lipid components;
5. acquisition-length handling and normalization;
6. optional lipid projection inputs for Y-Net models.

The output is a training batch containing:

```text
network_input   contaminated spectrum
network_target  nuisance component to remove
network_l2      optional lipid-projected input for Y-Net
```

The trained model predicts nuisance signal. Corrected spectra are obtained by subtracting the predicted nuisance component from the input spectrum/FID representation used by inference.

## Acquisition length

The simulation config controls whether all spectra use a fixed number of FID points or whether the acquired length is sampled. The training code supports fixed training pools and variable-length augmentation, but the model, simulator, and inference preprocessing must agree on the expected length range.

Relevant concepts:

- `min_acquired_n_timepoints`
- `max_acquired_n_timepoints`
- zero filling versus native-length cropping
- fixed training pools generated at maximum length

If a dataset has a different native FID length, resample/crop it before inference or use a model/config trained for that length.

## Water and lipid resources

Water/lipid resources must be prepared before an on-the-fly simulator can be
built. They contain subject-specific isolated water and lipid components and,
when enabled, lipid projection operators. Start from the acquisition-specific
config:

```text
configs/Simulation/3T_water_lipid_v1.yaml
configs/Simulation/7T_water_lipid_v1.yaml
```

Set its dataset root, subject list, input/mask paths, acquisition parameters,
and desired resource version. Then run:

```bash
python scripts/SimulationScripts/extract_water_lipid.py \
  configs/Simulation/7T_water_lipid_v1.yaml
```

The default config layout writes one resource file per subject, for example:

```text
<subject>/TrainData/SimulationResources_water_lipid_v1.h5
```

The `data.resources` section of the training config must point to this
same filename/version. A missing or mismatched resource file prevents the
simulation system from being constructed.

The preparation utilities live in:

```text
scripts/SimulationScripts/
```

Useful entry points include:

```text
extract_water_lipid.py
export_prepared_basis.py
update_vol_lipid_operators_840.py
update_train_3t_lipid_operators.py
```

These scripts are partly project- and dataset-specific. Check paths in the YAML before running them.

## Testing the simulator

Fast unit tests run by default:

```bash
python -m pytest -q
```

The full simulator integration tests require CUDA and local resources:

```bash
python -m pytest -q -m integration
```

Use those when changing simulator internals, basis handling, or training-system construction.
