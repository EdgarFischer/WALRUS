# Training

Model training is configured through YAML files and launched through `scripts/trainingScripts/train.py`.

## Main configs

Common starting points:

```text
configs/Training/train_3T.yaml
configs/Training/train_7T.yaml
configs/Training/train_7T_UNet_32_6_fixed_trainingset_100ep.yaml
```

The training config points to a simulation config, model architecture, output directory, optimizer settings, batch size, validation setup, and optional warm-start checkpoint.

## Launching training

From the repository root:

```bash
python scripts/trainingScripts/train.py \
  --config configs/Training/train_7T.yaml
```

Convenience shell wrappers are available:

```text
scripts/trainingScripts/train_3T.sh
scripts/trainingScripts/train_7T.sh
scripts/trainingScripts/train_all.sh
```

## Architectures

The training loop recognizes:

```text
unet
ynet
```

Conceptually:

- U-Net models receive the contaminated input spectrum.
- Y-Net models additionally require an L2/lipid-projected input branch.

Depth, channel count, and related architectural details are configured in YAML. The repository therefore does not treat "shallow" and "deep" as fixed model classes; they are experiment-specific descriptions of particular configurations.

## On-the-fly versus fixed training data

The default WALRUS training strategy uses on-the-fly simulation. This samples new spectra throughout training and generally improves generalization compared with a fixed precomputed pool.

Fixed training pools are also supported for controlled comparisons. In that mode:

- the pool is generated once;
- batches are sampled from the pool;
- optional acquisition-length augmentation crops all tensors consistently;
- the fixed pool should be stored at the maximum acquisition length.

The relevant implementation is in:

```text
src/walinet/training/training.py
```

## Warm starts

Warm-start training is configured by pointing a training YAML to a previous checkpoint. This is used for staged training such as 400 epochs followed by an additional 100 epochs.

When comparing warm-start runs, keep these constant unless intentionally varied:

- training subjects;
- validation subjects;
- simulator config;
- random seed policy;
- model architecture;
- validation set definition.

## Outputs

Training writes model checkpoints and logs to the configured model output directory. These outputs are local artifacts and are not versioned.

Typical files include:

```text
model_best.pt
model_last.pt
loss.txt
copied/expanded config files
```

## Tests

Fast tests:

```bash
python -m pytest -q
```

Integration tests with CUDA and simulator resources:

```bash
python -m pytest -q -m integration
```
