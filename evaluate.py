# module load gcc openmpi miniforge

import os

import numpy as np
import torch
import Pk_library as PKL

from tqdm import tqdm

import scipy.stats as stats
from typing import Dict, List, Tuple
import json
from metrics import ValidationSuite
import argparse
from utils import get_filepath, stats_dict

# python evaluate.py --model_folder ./run/wfm

# with N_grid=128 (Voxels per side), L_box = 25 Mpc/h, k_{N_{yq}} = (pi * 128) / 25 = 16.08 h/Mpc
# python evaluate.py --model_folder ./run/camels/wfm/ --start 900 --end 1000 --kmax 16.08 --boxsize 25
# python evaluate.py --model_folder ../IC_pixel-diffusion/run/cosmos_dm_1900_2 --unnormalize_sample
# python evaluate.py --model_folder ./run/wfm
# python evaluate.py --model_folder ./run/wfm_standard_64 --kmax 0.2 --target_type quijote_ic_64 --root ../Datasets
# python evaluate.py --model_folder ./run/wfm_standard_32 --kmax 0.1 --target_type quijote_ic_32 --root ../Datasets
# python evaluate.py --model_folder lightning_logs/version_7954002 --kmax 0.1 --target_type quijote_ic_32 --root ../Datasets
# python evaluate.py --model_folder ./run/wfm_lc_32 --kmax 0.1 --target_type lc_ic_32 --root ../Datasets --start 900 --end 910

# python evaluate.py --root ../Datasets --model_folder ./run/standard_32 --kmax 0.1 --target_type quijote_ic_32 --end 1907 --unnormalize_sample
# python evaluate.py --root ../Datasets --model_folder ./run/standard_64 --kmax 0.2 --target_type quijote_ic_64 --end 1905 --unnormalize_sample

parser = argparse.ArgumentParser(
    description='Evaluate generated samples', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--root', type=str, default='Dataset')
parser.add_argument('--model_folder', type=str, default='./run/cosmos_dm_1900_2')
parser.add_argument('--boxsize', type=float, default=1000)
parser.add_argument('--target_type', type=str, default='z127')
parser.add_argument('--kmax', type=float, default=0.4)
parser.add_argument('--start', type=int, default=1900)
parser.add_argument('--end', type=int, default=2000)
parser.add_argument('--normalize_target', action='store_true', help='whether to normalize target value')
parser.add_argument('--normalize_sample', action='store_true', help='whether to normalize the predicted samples')
parser.add_argument('--unnormalize_sample', action='store_true', help='whether to normalize the predicted samples')

args = parser.parse_args()

root = args.root
model_folder = args.model_folder

global_mean, global_std = stats_dict[args.target_type]
val_suite = ValidationSuite(
    boxsize=args.boxsize, kmax=args.kmax
)

for i, sample_no in tqdm(enumerate(range(args.start, args.end))):
    sample_folder = os.path.join(model_folder, 'samples', str(sample_no))
    samples_phys = np.load(os.path.join(sample_folder, 'sample.npy')).squeeze()
    truth_phys = np.load(os.path.join(root, get_filepath(sample_no, args.target_type)))

    # if the model outputs normalized samples instead of physical magnitude
    # this is only for Legin et al. who local normalizes truth
    # hence using local stats to unnormalize should give the predicted physics
    if args.unnormalize_sample:
        samples_phys = samples_phys * global_std + global_mean

    if args.normalize_sample:
        samples_global = (samples_phys - global_mean) / global_std
    else: samples_global = None 
    
    if args.normalize_target:
        truth_global = (truth_phys - global_mean) / global_std
    else: 
        truth_global = None 

    # Add to running statistics
    val_suite.add_example(samples_phys, truth_phys, 
                        samples_global, truth_global)
    
    # Can check progress at any time
    if (i + 1) % 10 == 0 and (sample_no+1) < args.end:
        current = val_suite._get_current_stats()
        print(f"After {i+1} examples: {current}")

results = val_suite._finalize_stats()

val_suite.print_summary(results)
val_suite.save_results(results, os.path.join(model_folder, 'results.json'))