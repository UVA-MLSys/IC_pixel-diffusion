import os
import numpy as np
from readfof import FoF_catalog
import MAS_library as MASL
import redshift_space_library as RSL

# --- Settings ---
BoxSize = 1000.0  # Mpc/h
grid = 128
Hubble = 100.0
redshift = 0.0
snapnum = 4
mass_cut = 1.32e13  # Msun/h
axis = 2  # Redshift distortion axis (z)
input_dir = "/scratch/dye7jx/Dataset/halos_lh"
output_dir = "/scratch/dye7jx/Projects/ICdiffusion/Dataset/halo_LH_128"
os.makedirs(output_dir, exist_ok=True)

# --- Loop over simulations ---
for sim_id in range(2000):
    print(f"Processing sim {sim_id}...")

    try:
        snapdir = os.path.join(input_dir, str(sim_id))
        fof = FoF_catalog(snapdir, snapnum, long_ids=False, swap=False, SFR=False, read_IDs=False)

        pos = fof.GroupPos / 1e3  # Convert from kpc/h to Mpc/h
        vel = fof.GroupVel * (1 + redshift)
        mass = fof.GroupMass * 1e10  # Convert to Msun/h

        # Mass cut
        mask = mass >= mass_cut
        pos = pos[mask]
        vel = vel[mask]

        # Apply redshift-space distortion
        RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

        # Assign to grid using PCS
        delta = np.zeros((grid, grid, grid), dtype=np.float32)
        MASL.MA(pos, delta, BoxSize, 'PCS', verbose=False)

        # Normalize to overdensity
        delta /= np.mean(delta, dtype=np.float32)
        delta -= 1.0

        # Save
        out_path = os.path.join(output_dir, f"halo_lh_{sim_id:04d}.npy")
        np.save(out_path, delta)

    except Exception as e:
        print(f"❌ Failed on sim {sim_id}: {e}")
