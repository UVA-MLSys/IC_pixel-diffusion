#!/bin/bash
#SBATCH --job-name=dm_1900
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --output=logs/dm_1900_%j.out
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
# torchrun \
#   --standalone \
#   --nproc_per_node=4 \
#   --master_port ${MASTER_PORT} \
#    train_complete.py --disable_tqdm --config config_dm_1900_2.json
torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port ${MASTER_PORT} \
   train_complete.py --disable_tqdm --config ./configs/config_camels.json

echo "Training completed at: $(date)"

# python train_complete.py --num_workers 0 --config ./configs/standard_32.json --disable_ddp