import os
import numpy as np
import shutil

# === Paths ===
base_dir = "/scratch/dye7jx/Dataset/Quijote_Obs"
target_dir = "/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z0_1900"
os.makedirs(target_dir, exist_ok=True)

# === Selected sim IDs ===
selected = 
selected_str = [f"{sim_id:03d}" for sim_id in selected]  # zero-padded

# === Save selected folder list ===
np.save("/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z0_selected_folders_4.npy", selected_str)

# === Copy df_m_128_z=0.npy from each folder ===
for folder in selected_str:
    src = os.path.join(base_dir, folder, "df_m_128_z=0.npy")
    dst = os.path.join(target_dir, f"{folder}_z0.npy")

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copied: {folder}")
    else:
        print(f"⚠️ Missing: {src}")
