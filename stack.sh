#!/bin/bash
#SBATCH --job-name=stack
#SBATCH --account=bii_dsc_community
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --mem=512G
#SBATCH --time=0:10:00
#SBATCH --output=logs/stack_100_%j.out

source /etc/profile.d/modules.sh
source ~/.bashrc

module load miniforge
conda activate astroclip

python Dataset/stack_sample.py