#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/Tumor_Patients/Tumor_Patients_2026"

DST_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/data/7T/NoB0Correction"

subjects=(
    "MS_180"
    # "MS_210"
    # "MS_230"
    # "MS_250"
    # "MS_260"
    # "MS_280"
    # "MS_320"
    # "MS_340"
    # "MS_400"
    # "MS_430"
)

SRC_SUBDIR="3D-CRT_m64x64x33_0p2_3p88ppm_woL2"

for subject in "${subjects[@]}"; do

    src_dir="${SRC_BASE}/${subject}/${SRC_SUBDIR}"
    dst_dir="${DST_BASE}/${subject}/OriginalData"

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

    mkdir -p "${dst_dir}"

    cp "${src_dir}/CombinedCSI.mat" "${dst_dir}/"
    cp -r "${src_dir}/maps" "${dst_dir}/"

    echo "Successfully finished ${subject}"
done

echo "Done."