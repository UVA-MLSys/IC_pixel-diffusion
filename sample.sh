#!/bin/bash
#SBATCH --job-name=sample_ddpm
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sample_ddpm_%j.out

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

# python sample.py --disable_tqdm --config config_dm_1900_2.json
# python batch_sample.py --disable_tqdm --start 1999 --end 2000

# 1 hour
python batch_sample.py --config ./configs/standard_32.json --disable_tqdm

# 7 hours
python batch_sample.py --config ./configs/standard_64.json --disable_tqdm