import os
import numpy as np
import readgadget
import MAS_library as MASL
import redshift_space_library as RSL

# === Parameters ===
grid = 128
MAS = 'CIC'
axis = 0  # Redshift-space distortion along x-axis
verbose = True
snap_template = "/scratch/dye7jx/Dataset/Latin_hypercube_snappdir_oo4/{i}/"
save_dir = "/scratch/dye7jx/Projects/ICdiffusion/Dataset/z0_redshift_density_128_1900"
os.makedirs(save_dir, exist_ok=True)

for i in range(48, 100):  # ⬅️ Updated range
    snapshot = snap_template.format(i=i)

    print(f"✅ Processing simulation {i}: {snapshot}")

    # Read header
    header = readgadget.header(snapshot)
    BoxSize = header.boxsize / 1e3  # Mpc/h
    Masses = header.massarr * 1e10  # Msun/h
    Omega_m = header.omega_m
    Omega_l = header.omega_l
    h = header.hubble
    redshift = header.redshift
    Hubble = 100.0 * np.sqrt(Omega_m * (1.0 + redshift)**3 + Omega_l)

    # Read CDM particle positions and velocities
    pos_c = readgadget.read_block(snapshot, "POS ", [1]) / 1e3
    vel_c = readgadget.read_block(snapshot, "VEL ", [1])

    # Move to redshift space
    RSL.pos_redshift_space(pos_c, vel_c, BoxSize, Hubble, redshift, axis)

    # Create density field
    delta = np.zeros((grid, grid, grid), dtype=np.float32)
    mass_c = np.ones(pos_c.shape[0], dtype=np.float32) * Masses[1]
    MASL.MA(pos_c, delta, BoxSize, MAS, W=mass_c, verbose=verbose)

    # Convert to overdensity
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0

    # Save
    np.save(f"{save_dir}/z0_redshift_{i:03d}.npy", delta)
    print(f"💾 Saved: {save_dir}/z0_redshift_{i:03d}.npy")
