#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

# -----------------------------------------------------------------------------
# Edit these three variables.
# SOURCES and DATASET_NAMES must contain the same number of entries.
# -----------------------------------------------------------------------------

TARGET_ROOT="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/WALRUS/data/10.5T"

SOURCES=(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/CooperationPartners/Gosia_Minnesota/Results/Measurement26/mrsi_x26_stability_CRT2_1"
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/CooperationPartners/Gosia_Minnesota/Results/Measurement29/mrsi_x29_stability_CRT2_64x64x39"
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/CooperationPartners/Gosia_Minnesota/Results/Measurement31/csi_fidesi_crt_6thNov024_3TI_2778"
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/CooperationPartners/Gosia_Minnesota/Results/Measurement34/csi_fidesi_CRT2_44x44x27"
)

DATASET_NAMES=(
    "Measurement26"
    "Measurement29"
    "Measurement31"
    "Measurement34"
)

# -----------------------------------------------------------------------------

REQUIRED_FILES=(
    "CombinedCSI.mat"
    "maps/mask_lipid.nii.gz"
    "maps/mask.nii.gz"
    "maps/magnitude.nii.gz"
)

if [[ ${#SOURCES[@]} -ne ${#DATASET_NAMES[@]} ]]; then
    echo "ERROR: SOURCES and DATASET_NAMES must have the same length." >&2
    echo "SOURCES:       ${#SOURCES[@]}" >&2
    echo "DATASET_NAMES: ${#DATASET_NAMES[@]}" >&2
    exit 1
fi

if [[ ${#SOURCES[@]} -eq 0 ]]; then
    echo "ERROR: No source directories configured." >&2
    exit 1
fi

# Validate everything before creating or modifying destination directories.
for index in "${!SOURCES[@]}"; do
    source_dir=${SOURCES[$index]}
    dataset_name=${DATASET_NAMES[$index]}

    if [[ -z "$dataset_name" || "$dataset_name" == */* ]]; then
        echo "ERROR: Invalid dataset name: '$dataset_name'" >&2
        echo "Use a single directory name without '/'." >&2
        exit 1
    fi

    if [[ ! -d "$source_dir" ]]; then
        echo "ERROR: Source directory does not exist: $source_dir" >&2
        exit 1
    fi

    for relative_path in "${REQUIRED_FILES[@]}"; do
        if [[ ! -f "$source_dir/$relative_path" ]]; then
            echo "ERROR: Missing required file: $source_dir/$relative_path" >&2
            exit 1
        fi
    done
done

mkdir -p "$TARGET_ROOT"

for index in "${!SOURCES[@]}"; do
    source_dir=${SOURCES[$index]}
    dataset_name=${DATASET_NAMES[$index]}
    original_data_dir="$TARGET_ROOT/$dataset_name/OriginalData"
    maps_dir="$original_data_dir/maps"

    echo "[$((index + 1))/${#SOURCES[@]}] Copying $source_dir"
    echo "    -> $original_data_dir"

    mkdir -p "$maps_dir"

    cp -a "$source_dir/CombinedCSI.mat" "$original_data_dir/CombinedCSI.mat"
    cp -a "$source_dir/maps/mask_lipid.nii.gz" "$maps_dir/mask_lipid.nii.gz"
    cp -a "$source_dir/maps/mask.nii.gz" "$maps_dir/mask.nii.gz"
    cp -a "$source_dir/maps/magnitude.nii.gz" "$maps_dir/magnitude.nii.gz"
done

echo "Finished copying ${#SOURCES[@]} dataset(s) to: $TARGET_ROOT"
echo
echo "Extracting data.npy, consistently oriented masks, and QC plots..."

"$PYTHON_BIN" "$SCRIPT_DIR/extract_combined_csi_data_and_masks.py" \
    "$TARGET_ROOT" \
    --datasets "${DATASET_NAMES[@]}"

echo
echo "Finished copying and preparing all ${#SOURCES[@]} dataset(s)."
