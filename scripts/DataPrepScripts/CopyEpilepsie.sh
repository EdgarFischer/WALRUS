#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/Epilepsy_Patients/EPI_0002"

DST_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/WALRUS/data/7T/NoB0Correction"

subjects=(
    "EPI_0002_2026-11-07_v1.0"
)

map_files=(
    "magnitude.nii"
    "mask_lipid.nii"
    "mask.nii"
)

for subject in "${subjects[@]}"; do

    src_dir="${SRC_BASE}/${subject}"
    dst_dir="${DST_BASE}/${subject}/OriginalData"
    dst_maps_dir="${dst_dir}/maps"

    echo "Processing ${subject}"

    if [[ ! -d "${src_dir}" ]]; then
        echo "ERROR: Source directory does not exist: ${src_dir}" >&2
        exit 1
    fi

    if [[ ! -f "${src_dir}/CombinedCSI.mat" ]]; then
        echo "ERROR: Missing CombinedCSI.mat: ${src_dir}/CombinedCSI.mat" >&2
        exit 1
    fi

    if [[ ! -d "${src_dir}/maps" ]]; then
        echo "ERROR: Missing maps directory: ${src_dir}/maps" >&2
        exit 1
    fi

    for file in "${map_files[@]}"; do
        if [[ ! -f "${src_dir}/maps/${file}" ]]; then
            echo "ERROR: Missing map file: ${src_dir}/maps/${file}" >&2
            exit 1
        fi
    done

    mkdir -p "${dst_maps_dir}"

    cp "${src_dir}/CombinedCSI.mat" "${dst_dir}/"

    for file in "${map_files[@]}"; do
        cp "${src_dir}/maps/${file}" "${dst_maps_dir}/"
    done

    echo "Successfully finished ${subject}"
done

echo "Done."