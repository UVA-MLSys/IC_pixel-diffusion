#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/sample_1900_%j.out
#SBATCH --mail-user=mi3se@virginia.edu
#SBATCH --mail-type=END,FAIL

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

# python sample.py --disable_tqdm --config config_dm_1900_2.json
python batch_sample.py --disable_tqdm --start 1999 --end 2000