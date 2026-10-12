# Data formats

WALRUS works with complex-valued MRSI data in which the final axis is the time-domain FID. Most workflows assume NumPy arrays on disk, while some conversion and inference utilities also support `CombinedCSI.mat`.

## Subject layout

A prepared subject should usually look like:

```text
<base_dir>/<Subject>/
├── OriginalData/
│   ├── data.npy
│   ├── data_after_walrus.npy        # produced by inference
│   └── CombinedCSI.mat              # optional source/interop format
└── masks/
    ├── brain_mask.npy
    └── lipid_mask.npy               # needed for Y-Net / lipid projection workflows
```

The exact `<base_dir>` is configured in YAML files. Common examples are 3T and 7T local dataset trees.

## `data.npy`

Expected shape:

```text
(nx, ny, nz, nt)
```

where:

- `nx, ny, nz` are spatial CSI dimensions;
- `nt` is the number of FID points;
- values are complex, typically `complex64` or `complex128`;
- the final axis is time, not frequency.

Typical examples:

```text
7T: (64, 64, 35, 840)
7T: (64, 64, 39, 840)
3T: site/acquisition dependent
```

If source data have a longer FID, crop/resample before training or inference so that the model and acquisition config agree.

## Masks

Expected shape:

```text
(nx, ny, nz)
```

Masks should be aligned voxel-for-voxel with `data.npy`.

- `brain_mask.npy`: voxels used for inference, fitting, or evaluation.
- `lipid_mask.npy`: scalp/lipid voxels used by lipid projection and Y-Net workflows.

Masks are treated as binary after thresholding/nonzero conversion. Use consistent orientation before training or inference; do not rely on later visualization code to fix orientation.

## `CombinedCSI.mat`

Some data-preparation and inference scripts read or write MATLAB-style CSI containers. The commonly expected fields are:

```text
csi.Data      complex CSI array, usually (x, y, z, t)
mask          brain mask or related binary mask
```

Capitalization can vary across source files, so the I/O helpers in `walinet.data.combined_csi_io` are preferred over ad-hoc loading.

## Model outputs

Inference typically writes:

```text
OriginalData/data_after_walrus.npy
```

This has the same spatial dimensions and FID length as the model output. For LCModel or downstream fitting, ensure that the fit script uses the correct acquisition length and does not silently assume a different number of FID points.

## NIfTI maps

LCModel-derived metabolite maps are usually written as `.nii.gz` under fit-specific folders. When creating additional NIfTI maps from `data.npy`, use a reference map from the same fit branch so that Freeview/FSL/Inkscape see compatible headers and affine transforms.

## What is not versioned

Large local data products are intentionally ignored by Git, including generated results, paper graphics, metabolite-mode resources, and nested notebook workspaces. The repository documents the expected structure but does not distribute these files.
