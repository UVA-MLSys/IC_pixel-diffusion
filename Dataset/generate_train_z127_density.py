# Generates 3D density fields at z=127 for selected Quijote Latin Hypercube simulations
from mpi4py import MPI
import numpy as np
import os
import readgadget
import MAS_library as MASL

###### MPI DEFINITIONS ######
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# === CONFIG ===
grid         = 128
z            = 127
ptypes       = [1]
input_ids_path = 
output_dir   = '/scratch/dye7jx/Projects/ICdiffusion/Dataset/Train_z127_from_IC_1900'

# === FUNCTION ===
def compute_df(snapshot, ptypes, grid, fout):
    if not os.path.exists(snapshot + '.0.hdf5'):
        print(f'🚫 Snapshot not found: {snapshot}.0.hdf5')
        return
    df = MASL.density_field_gadget(snapshot, ptypes, grid, MAS='CIC',
                                   do_RSD=False, axis=0, verbose=True)
    df = df / np.mean(df, dtype=np.float64) - 1.0  # overdensity
    np.save(fout, df)
    print(f'✅ Saved density field to: {fout}')

# === MAIN ===
if myrank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
comm.Barrier()

# Load simulation IDs
input_ids = np.load(input_ids_path)

# Distribute work among MPI ranks
my_ids = input_ids[myrank::nprocs]

for sim_id in my_ids:
    snapshot = f'/scratch/dye7jx/Dataset/Quijote_ICs/{sim_id}/ICs/ics'
    fout = os.path.join(output_dir, f'df_m_z={z}_sim{sim_id}.npy')
    compute_df(snapshot, ptypes, grid, fout)

comm.Barrier()
