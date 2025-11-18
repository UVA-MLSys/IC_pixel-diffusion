#!/usr/bin/env python3
import os
import numpy as np

# === Paths ===
z0_dir   = "./Dataset/Train_z0_2000"
z127_dir = "./Dataset/Train_z127_from_IC_2000"
halo_dir = "./Dataset/halo_LH_128"
out_dir  = "./Dataset"

sim_ids = list(range(1900))

z0_list, z127_list, halo_list, missing = [], [], [], []

for sim in sim_ids:
    sim_str = f"{sim:04d}"                               # zero-padded for z0
    f_z0    = os.path.join(z0_dir,   f"{sim}_z0.npy")
    f_z127  = os.path.join(z127_dir, f"df_m_z=127_sim{sim}.npy")  # no padding
    f_halo = os.path.join(halo_dir, f'halo_lh_{sim_str}.npy')

    ok0, ok127, ok_halo = os.path.exists(f_z0), os.path.exists(f_z127), os.path.exists(f_halo)
    if not (ok0 and ok127 and ok_halo):
        print(f"Missing pair for sim {sim_str} -> z0={ok0} z127={ok127}, halo {ok_halo}")
        if not ok0:   print(" expected z0:  ", f_z0)
        if not ok127: print(" expected z127:", f_z127)
        if not ok_halo: print(f" expected halo: {f_halo}")
        missing.append(sim)
        continue

    z0   = np.load(f_z0).astype(np.float32)     # expect (128,128,128) or (1,128,128,128)
    z127 = np.load(f_z127).astype(np.float32)
    halo = np.load(f_halo).astype(np.float32)

    # If any file has a leading singleton channel, drop it
    if z0.ndim == 4 and z0.shape[0] == 1:       # (1,D,H,W) -> (D,H,W)
        z0 = z0[0]
    if z127.ndim == 4 and z127.shape[0] == 1:
        z127 = z127[0]
    if halo.ndim == 4 and halo.shape[0] == 1:
        halo = halo[0]

    if z0.shape != z127.shape:
        raise ValueError(f"Shape mismatch for sim {sim_str}: z0 {z0.shape} vs z127 {z127.shape}")

    if z0.ndim != 3:
        raise ValueError(f"Unexpected ndim for sim {sim_str}: got {z0.ndim}, expected 3 (D,H,W)")


    z0_list.append(z0)
    z127_list.append(z127)
    halo_list.append(halo)

print("Loaded pairs:", len(z0_list))
if not (z0_list and z127_list and halo_list):
    raise SystemExit("No valid pairs to stack. Check filenames/IDs.")

# Stack -> (N, D, H, W)
z0_stack   = np.stack(z0_list,   axis=0)
z127_stack = np.stack(z127_list, axis=0)
halo_stack = np.stack(halo_list, axis=0)

print("Final shapes:", z0_stack.shape, z127_stack.shape, halo_stack.shape)  # should be (N,128,128,128)

os.makedirs(out_dir, exist_ok=True)
out_z0   = os.path.join(out_dir,   f"quijote128_dm_train_{z0_stack.shape[0]}.npy")
out_z127 = os.path.join(out_dir, f"quijote128_z127_train_{z127_stack.shape[0]}.npy")
out_halo = os.path.join(out_dir, f"quijote128_halo_train_{halo_stack.shape[0]}.npy")

np.save(out_z0,   z0_stack)
np.save(out_z127, z127_stack)
np.save(out_halo, halo_stack)

# Quick mmap sanity
z0_m   = np.load(out_z0, mmap_mode="r")
z127_m = np.load(out_z127, mmap_mode="r")
halo_m = np.load(out_halo, mmap_mode="r")

print("Saved:")
print(" ", out_z0,   "->", z0_m.shape,   z0_m.dtype)
print(" ", out_z127, "->", z127_m.shape, z127_m.dtype)
print(" ", out_halo, "->", halo_m.shape, halo_m.dtype)