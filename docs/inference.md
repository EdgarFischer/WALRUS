# Inference

Inference applies a trained model to complex MRSI FIDs and writes corrected data for downstream fitting or visualization.

## Core implementation

The reusable inference code lives in:

```text
src/walinet/inference/fid_inference.py
src/walinet/inference/prediction.py
src/walinet/inference/ford_pipeline.py
```

Project-specific entry points live in:

```text
scripts/inference/
```

## Typical inputs

For NumPy-based inference:

```text
OriginalData/data.npy
masks/brain_mask.npy
```

For Y-Net / L2-prior inference, a lipid projection operator or equivalent lipid information is also required. Single-input U-Net models do not require the Y-Net projection branch.

## Typical outputs

The usual corrected output is:

```text
OriginalData/data_after_walrus.npy
```

Some scripts write model-specific names, for example:

```text
data_WALINET_Style.npy
data_after_walrus_420.npy
data_after_walrus_630.npy
```

Use explicit names when comparing models or FID lengths.

## Running inference scripts

Example pattern:

```bash
python scripts/inference/infer_ucsf_7t_b0corrected.py
```

Many scripts are dataset-specific. Before running one, check:

- input data path;
- brain mask path;
- model directory;
- checkpoint name;
- device/GPU;
- output path;
- expected FID length.

For urgent/manual GPU runs, prefer setting `CUDA_VISIBLE_DEVICES` or editing the script device explicitly rather than relying on defaults.

## FID length handling

Inference must match the model’s training acquisition:

- zero-filled models expect a fixed length;
- variable-length models may accept a range but still obey configured min/max values;
- data with native lengths outside the range must be cropped, padded, or resampled first.

Do not assume LCModel fit scripts can handle arbitrary FID length without checking the corresponding control file and acquisition parameters.

## CombinedCSI support

Some workflows use `CombinedCSI.mat`. Use the helpers in:

```text
src/walinet/data/combined_csi_io.py
```

rather than hand-coding field access, because source files may differ in field naming and storage conventions.

## Quality control

Recommended QC after inference:

1. confirm input/output shapes;
2. inspect representative spectra before and after correction;
3. verify orientation against brain masks and magnitude/anatomical references;
4. run LCModel/fitting only after the corrected FIDs and masks are confirmed consistent.

For paper figures, prefer fixed selected voxels and saved selection files so that examples are reproducible.
