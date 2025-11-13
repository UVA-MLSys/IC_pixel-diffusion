import os
import numpy as np
from utils import get_config

# === Settings ===
N = 128
noise_sigma = 0.1  # Should match training config

# === File paths ===
z127_path = f"./Dataset/Train_z127_from_IC_2000/df_m_z=127_sim1999.npy"
z0_path = f"./Dataset/Train_z0_2000/1999_z0.npy"

config = get_config('./config.json')
output_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# === Load z=0 and add Gaussian noise ===
z0 = np.load(z0_path).reshape(N, N, N)
z0_noisy = z0 + noise_sigma * np.random.normal(size=z0.shape)
z0_noisy = z0_noisy[np.newaxis, ...]  # shape: (1, 128, 128, 128)

# === Load z=127 and normalize ===
z127 = np.load(z127_path).reshape(N, N, N)
z127_norm = (z127 - np.mean(z127)) / np.std(z127)
z127_norm = z127_norm[np.newaxis, ...]

# === Save as observation and truth ===
np.save(os.path.join(output_dir, "observation.npy"), z0_noisy)
np.save(os.path.join(output_dir, "truth.npy"), z127_norm)

print(f"✅ Saved observation and truth for sim 9 to {output_dir}")
