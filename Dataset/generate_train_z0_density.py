import numpy as np
import os
import MAS_library as MASL

# === Settings ===
simulation_ids = list(range(0, 2000))  # 0..400 inclusive
base_path = '/scratch/dye7jx/Dataset/Latin_hypercube_snappdir_oo4'
output_dir = '/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z0_2000_512'
grid = 512
BoxSize = 1000.0  # Mpc/h
ptypes = [1]  # dark matter particles

os.makedirs(output_dir, exist_ok=True)

def compute_density_field(sim_id):
    snapshot = f'{base_path}/{sim_id}/snap_004'
    hdf5_file = snapshot + '.0.hdf5'
    if not os.path.exists(hdf5_file):
        print(f"❌ File not found: {hdf5_file}")
        return

    print(f"✅ Processing z=0 for simulation {sim_id}")
    df = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='PCS',
                                   do_RSD=False, axis=0, verbose=False)
    df = df / np.mean(df, dtype=np.float64) - 1.0  # Overdensity normalization
    np.save(f'{output_dir}/z0_{sim_id:04d}.npy', df)  # <-- unique file per sim

# === Run ===
for sim_id in simulation_ids:
    compute_density_field(sim_id)
