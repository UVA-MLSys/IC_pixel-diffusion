#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sample_100_%j.out

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

python sample.py --disable_tqdm --config config_halo_100.json