#!/bin/bash
#SBATCH --job-name=sample_ddpm_standard_128
#SBATCH --account=bii_dsc_community
#SBATCH --partition=gpu
#---SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=logs/sample_ddpm_standard_128_%j.out

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn miniforge
conda activate astroclip

# python sample.py --disable_tqdm --config config_dm_1900_2.json
# python batch_sample.py --disable_tqdm --start 1999 --end 2000

# # 1 hour
# python batch_sample.py --config ./configs/standard_32.json --disable_tqdm

# # 7 hours
# python batch_sample.py --config ./configs/standard_64.json --disable_tqdm

python batch_sample_ddim.py --config ./configs/standard_128.json --disable_tqdm

# python batch_sample.py --config ./configs/lc_128.json --start 900 --end 1000 --disable_tqdm
# python batch_sample.py --config ./configs/lc_32.json --start 900 --end 1000 --disable_tqdm
# python batch_sample.py --config ./configs/lc_64.json --start 900 --end 1000 --disable_tqdm

# python batch_sample.py --config ./configs/bsq_128.json --start 900 --end 1000 --disable_tqdm
# python batch_sample.py --config ./configs/bsq_32.json --start 900 --end 1000 --disable_tqdm
# python batch_sample.py --config ./configs/bsq_64.json --start 900 --end 1000 --disable_tqdm