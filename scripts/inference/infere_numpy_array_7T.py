import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from walinet.inference.fid_inference import infer_fid

subject = "UCSF/Vol3"
model = "UNet_ablation_WALINET_Calibration"
file_name = "data_after_U20_WALINET_calibration.npy"

infer_fid(
    fid=f"/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/{subject}/OriginalData/data.npy",
    headmask=f"/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/{subject}/masks/brain_mask.npy",
    model_dir=f"/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/models/{model}",
    output_path=f"/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/{subject}/OriginalData/{file_name}",
    fid_axis="auto",
    device="cuda:0",
)