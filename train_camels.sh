#!/bin/bash
#SBATCH --job-name=dm_camels
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=5
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --output=logs/dm_camels_%j.out

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

torchrun \
  --standalone \
  --nproc_per_node=2 \
  --master_port ${MASTER_PORT} \
   train_complete.py --disable_tqdm \
   --config ./configs/config_camels.json \
   --num_workers 3


echo "Training completed at: $(date)"