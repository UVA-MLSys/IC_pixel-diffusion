#!/usr/bin/env python3
import os
import numpy as np

# === File paths ===
z0_dir   = "/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z0_2000"
z127_dir = "/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z127_from_IC_2000"

sim_ids = list(range(1900))

z0_data = []
z127_data = []
missing = []

for sim in sim_ids:
    sim_str = f"{sim:04d}"  # zero-padded for z0 filenames
    file_z0   = os.path.join(z0_dir,   f"{sim}_z0.npy")
    file_z127 = os.path.join(z127_dir, f"df_m_z=127_sim{sim}.npy")

    if not (os.path.exists(file_z0) and os.path.exists(file_z127)):
        print(f"⚠️ Missing pair for sim {sim} -> z0={os.path.exists(file_z0)} z127={os.path.exists(file_z127)}")
        missing.append(sim)
        continue

    # Load as float32 to keep memory predictable
    z0   = np.load(file_z0).astype(np.float32)     # (C,D,H,W)
    z127 = np.load(file_z127).astype(np.float32)   # (C,D,H,W)

    z0_data.append(z0)
    z127_data.append(z127)

print("🧪 Done loading. Checking lengths...")
print("🔍 len(z0_data):", len(z0_data))
print("🔍 len(z127_data):", len(z127_data))

if missing:
    print(f"⚠️ Missing {len(missing)} sims (showing up to 20): {missing[:20]}")

if len(z0_data) > 0 and len(z127_data) > 0:
    print("📦 Starting stacking...")
    try:
        # This will allocate ~4GB per array (240 × 128 × 32 × 32 × 32 × 4 bytes)
        z0_array   = np.stack(z0_data,   axis=0)   # (N, C, D, H, W)
        z127_array = np.stack(z127_data, axis=0)   # (N, C, D, H, W)
        print("✅ Stacking successful.")

        out_dir = "/scratch/dye7jx/Projects/ICdiffusion/Dataset"
        os.makedirs(out_dir, exist_ok=True)

        out_z0   = os.path.join(out_dir, "quijote128_z0_train_1950_comb.npy")
        out_z127 = os.path.join(out_dir, "quijote128_z127_train_1900_comb.npy")
        np.save(out_z0,   z0_array)
        np.save(out_z127, z127_array)

        print("✅ Arrays saved successfully.")
        print("📄", out_z0)
        print("📄", out_z127)

        # --- quick sanity check (does not load whole arrays into RAM thanks to mmap) ---
        z0_m   = np.load(out_z0, mmap_mode="r")
        z127_m = np.load(out_z127, mmap_mode="r")
        print("🔎 z0 stack:",   z0_m.shape,   z0_m.dtype)
        print("🔎 z127 stack:", z127_m.shape, z127_m.dtype)
        # Light stats on a small prefix to avoid big RAM use
        print("📊 z0 first-4 mean/std:",   float(z0_m[:4].mean()),   float(z0_m[:4].std()))
        print("📊 z127 first-4 mean/std:", float(z127_m[:4].mean()), float(z127_m[:4].std()))

        print("🔚 End of script reached.")
    except Exception as e:
        print("❌ Error during stacking/saving:", e)
else:
    print("⚠️ Something went wrong. One or both lists are empty.")
    print("🔚 End of script.")
