#!/bin/bash
#SBATCH --job-name=halo_100
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                # one Slurm task; torchrun will spawn 4 procs
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/halo_100_%j.out
#---SBATCH --mail-user=mi3se@virginia.edu
#---SBATCH --mail-type=END,FAIL

module load gcc nccl

# === Environment ===
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

# === Debug info ===
mkdir -p logs
echo "Node: $SLURMD_NODENAME"
nvidia-smi

# Pick a port unlikely to collide (change if needed)
export MASTER_PORT=29602
# Single node: MASTER_ADDR can be localhost
export MASTER_ADDR=127.0.0.1

echo "Starting DDP training with torchrun at: $(date)"

# === Launch (single node, 4 GPUs) ===
torchrun \
  --standalone \
  --nproc_per_node=1 \
  --master_port ${MASTER_PORT} \
   train_complete.py --disable_tqdm --config config_halo_100_2.json

echo "Training completed at: $(date)"