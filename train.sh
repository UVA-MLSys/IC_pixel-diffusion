#!/bin/bash
#SBATCH --job-name=ddpm_lc_128
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#---SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=logs/lc_128_%j.out

module load gcc nccl

# === Environment ===
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

# python train_complete.py --num_workers 2 --config ./configs/standard_32.json --disable_ddp --disable_tqdm
# python train_complete.py --num_workers 2 --config ./configs/standard_64.json --disable_ddp --disable_tqdm

# python train_complete.py --num_workers 2 --config ./configs/lc_32.json --disable_ddp --disable_tqdm
# python train_complete.py --num_workers 2 --config ./configs/lc_64.json --disable_ddp --disable_tqdm
python train_complete.py --num_workers 3 --config ./configs/lc_128.json --disable_ddp --disable_tqdm

# python train_complete.py --num_workers 2 --config ./configs/bsq_128.json --disable_ddp --disable_tqdm
# python train_complete.py --num_workers 2 --config ./configs/bsq_64.json --disable_ddp --disable_tqdm

# python train_complete.py --num_workers 2 --config ./configs/bsq_64.json --disable_ddp --disable_tqdm