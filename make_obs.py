import os, sys
import numpy as np
from utils import get_config

# === File paths ===

sample_no = 1999

config = get_config(str(sys.argv[1])) # './config.json'

z127_path = f"./Dataset/Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
# z127_path = f"../IC-Flow-Diffusion/Dataset/Train_z127_CAMELS/z127_{sample_no:04d}.npy"
# z0_path = f"../IC-Flow-Diffusion/Dataset/Train_z0_CAMELS/z0_{sample_no:04d}.npy"

if config.data.input_type == 'halo':
    z0_path = f"./Dataset/halo_LH_128/halo_lh_{sample_no}.npy" 
else:
    z0_path = f"./Dataset/Train_z0_2000/{sample_no}_z0.npy"

print(f'Reading {z127_path} as the target truth and {z0_path} as the input observation')
output_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# === Load z=0 and add Gaussian noise ===
N = config.data.image_size
z0 = np.load(z0_path).reshape(N, N, N)
noise_sigma = config.data.noise_sigma
z0_noisy = z0 + noise_sigma * np.random.normal(size=z0.shape)
z0_noisy = z0_noisy[np.newaxis, ...]  # shape: (1, 128, 128, 128)

# === Load z=127 and normalize ===
z127 = np.load(z127_path).reshape(N, N, N)
z127_norm = (z127 - np.mean(z127)) / np.std(z127)
z127_norm = z127_norm[np.newaxis, ...]

# === Save as observation and truth ===
np.save(os.path.join(output_dir, "observation.npy"), z0_noisy)
np.save(os.path.join(output_dir, "truth.npy"), z127_norm)

print(f"✅ Saved observation and truth to {output_dir}")
