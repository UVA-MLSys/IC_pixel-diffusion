# %%
import os, sys
import numpy as np
import csv
import time
from utils import get_config
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from utils import get_sigma_time, get_sample_time, VESDE_DDIM, get_config, get_filepath
from model import UNet3DModel
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
import logging
import os
import sys
import datetime
from os.path import join
import argparse

# 1h
# python batch_sample_ddim.py --config ./configs/standard_32.json

# 7 hours
# python batch_sample_ddim.py --config ./configs/standard_64.json

# 1-1.30h
# python batch_sample_ddim.py --config ./configs/bsq_32.json --start 900 --end 1000

# python batch_sample_ddim.py --config ./configs/bsq_32.json --start 900 --end 1000
# python batch_sample_ddim.py --config ./configs/bsq_64.json --start 900 --end 1000
# python batch_sample_ddim.py --config ./configs/bsq_128.json --start 900 --end 1000

# python batch_sample_ddim.py --config ./configs/lc_32.json --start 900 --end 1000
# python batch_sample_ddim.py --config ./configs/lc_64.json --start 900 --end 1000
# python batch_sample_ddim.py --config ./configs/lc_128.json --start 900 --end 1000

def get_parser():
    parser = argparse.ArgumentParser(
        description='Sample using Diffusion', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start', type=int, default=1900, help='Start sample no')
    parser.add_argument('--end', type=int, default=2000, help='End sample no')
    parser.add_argument('--config', type=str, default='./configs/config_dm_1900_2.json', help='Configuration file')
    parser.add_argument(
        '--disable_tqdm', action='store_true', 
        help='whether to enable tqdm progress bar'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Enable benchmarking'
    )
    
    return parser

parser = get_parser()
args = parser.parse_args()

# %%
config =get_config(args.config)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Define variables for benchmark snippets
eval_batch_size = config.sampling.batch_size
n_samples = config.sampling.num_samples # This is num_samples per simulation

if args.benchmark:
    logging.info("Benchmarking enabled.")
    world_size = 1 # sample.py is single-GPU
    model_folder_bench = join(config.model.workdir, config.model.cosmo_dir)
    benchmark_file = os.path.join(model_folder_bench, f"inference_benchmark_{world_size}_ddim.csv")
    logging.info(f"Benchmark results will be saved to: {benchmark_file}")
    fieldnames = [
        "eval_batch_size",
        "num_simulations",
        "num_samples_per_sim",
        "total_samples_generated",
        "total_time_s",
        "total_generate_time_s",
        "io_overhead_time_s",
        "generate_time_percent",
        "peak_memory_gb",
        "sec_per_sample",
    ]
    with open(benchmark_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

total_generate_time = 0.0

model_folder = join(config.model.workdir, config.model.cosmo_dir)
checkpoint_dir = join(model_folder, config.model.checkpoint_dir)
data_root = config.data.path

input_type = config.data.input_type

# %%
# Initialize score model
model = UNet3DModel(config)
#model = DataParallel(model)
model = model.to(DEVICE)

ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

inference_steps = 100  
schedule_method = "quadratic" # Matches your Keras implementation

# Initialize the updated SDE class
sde_ddim = VESDE_DDIM(
    config.model.sigma_min, config.model.sigma_max, 
    config.model.num_scales, config.model.T, config.model.sampling_eps
)

# Generate the schedule
# Timesteps go from T (noisy) down to eps (clean)
timesteps = sde_ddim.get_ddim_schedule(inference_steps + 1, method=schedule_method)
t_stimesteps = timesteps.to(DEVICE)

print(f'Sampling with DDIM ({schedule_method} schedule, {inference_steps} steps).')

# Check for existing checkpoint
checkpoint_path = join(checkpoint_dir, 'checkpoint.pth')
if os.path.isfile(checkpoint_path):
    loaded_state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])
    init_epoch = int(loaded_state['epoch'])
    logging.info(f"Loaded checkpoint from {checkpoint_path}.")
    print(f"Loaded checkpoint from {checkpoint_path}.")
else:
    logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

# --- Start benchmark timer and reset memory stats ---
if torch.cuda.is_available() and args.benchmark:
    torch.cuda.reset_peak_memory_stats('cuda:0')
start_time = time.time()

model.eval()

for sample_no in tqdm(range(args.start, args.end), disable=args.disable_tqdm):
    z0_path = os.path.join(data_root, get_filepath(sample_no, input_type)) # f"./Dataset/Train_z0_2000/{sample_no}_z0.npy"

    output_dir = os.path.join(
        model_folder, 'samples_ddim', str(sample_no)
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # === Load z=0 and add Gaussian noise ===
    N = config.data.image_size
    z0 = np.load(z0_path).reshape(N, N, N)
    noise_sigma = config.data.noise_sigma
    z0_noisy = z0 + noise_sigma * np.random.normal(size=z0.shape)
    z0_noisy = z0_noisy[np.newaxis, ...]  # shape: (1, 128, 128, 128)

    Nside = config.data.image_size
    #DEVICE = config.device

    sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
    sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

    input_data = torch.from_numpy(np.float32(z0_noisy)).to(DEVICE)
    input_data = torch.unsqueeze(input_data, dim=1)

    input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
    shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

    timesteps = sde_ddim.get_ddim_schedule(inference_steps + 1, method=schedule_method)

    samples = []
    for j in range(config.sampling.num_samples // config.sampling.batch_size):
        # --- Time the generation step specifically ---
        if args.benchmark:
            generate_start_time = time.time()

        with torch.no_grad(), ema.average_parameters():
            # Start with random noise
            x = sde_ddim.prior_sampling(shape).to(DEVICE)
            
            # Iterate through the schedule
            # We stop at len(timesteps) - 1 because we need a "next" step t_prev
            for i in range(len(timesteps) - 1):
                t = timesteps[i]
                t_prev = timesteps[i+1]
                
                # Broadcast time to batch size
                t_vec = torch.ones(shape[0], device=DEVICE) * t
                t_prev_vec = torch.ones(shape[0], device=DEVICE) * t_prev
                
                # Run model
                # Note: Ensure your model accepts inputs in this order
                model_output = model(torch.cat([x, input_data], dim=1), t_vec)
                
                # DDIM Update (eta=0.0 for deterministic)
                x, x_mean = sde_ddim.ddim_step(x, t_vec, t_prev_vec, model_output, eta=0.0)

            # Store results
            samples.append(x.detach().cpu().numpy()) #

        if args.benchmark:
            generate_end_time = time.time()
            total_generate_time += (generate_end_time - generate_start_time)
        
        # Save intermediate results
        np.save(join(output_dir, 'sample.npy'), np.array(samples))
        # print(f'Finished batch {j+1}')

    samples = np.array(samples).reshape(-1, Nside, Nside, Nside)
    np.save(os.path.join(output_dir, 'sample.npy'), samples) #

# --- End benchmark and log results ---
total_time = time.time() - start_time

if args.benchmark:
    peak_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)

    num_simulations = args.end - args.start
    total_generated_samples = num_simulations * n_samples
    sec_per_sample = total_time / total_generated_samples if total_generated_samples > 0 else 0.0

    io_overhead_time = total_time - total_generate_time
    generate_time_percent = (total_generate_time / total_time) * 100 if total_time > 0 else 0.0

    logging.info("--- Inference Benchmark Results ---")
    logging.info(f"Total input simulations: {num_simulations}")
    logging.info(f"Samples per simulation: {n_samples}")
    logging.info(f"Total samples generated: {total_generated_samples}")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"  - Total generate time: {total_generate_time:.2f}s ({generate_time_percent:.1f}%)")
    logging.info(f"  - I/O & overhead time: {io_overhead_time:.2f}s")
    logging.info(f"Peak memory: {peak_memory_gb:.2f} GB")
    logging.info(f"Time per sample: {sec_per_sample:.2f} s")

    with open(benchmark_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "eval_batch_size": eval_batch_size,
            "num_simulations": num_simulations,
            "num_samples_per_sim": n_samples,
            "total_samples_generated": total_generated_samples,
            "total_time_s": round(total_time, 4),
            "total_generate_time_s": round(total_generate_time, 4),
            "io_overhead_time_s": round(io_overhead_time, 4),
            "generate_time_percent": round(generate_time_percent, 2),
            "peak_memory_gb": round(peak_memory_gb, 4),
            "sec_per_sample": round(sec_per_sample, 4),
        })

    # --- Append to common benchmark file ---
    def get_original_cwd():
        return os.getcwd()

    common_benchmark_file = os.path.join(get_original_cwd(), "results", "inference_master.csv")
    os.makedirs(os.path.dirname(common_benchmark_file), exist_ok=True)
    
    common_fieldnames = [
        "timestamp", "model_folder", "model_name", "resolution",
        "world_size", "gpu_name", "eval_batch_size",
        "num_simulations", "num_samples_per_sim", "total_samples_generated",
        "total_time_s", "total_generate_time_s", "io_overhead_time_s", "generate_time_percent",
        "peak_memory_gb", "sec_per_sample",
    ]

    model_name = "Diffusion (DDIM)"
    
    resolution = 0
    for res_val in ["32", "64", "128", "256", "512", "1024"]:
        if res_val in config.data.target_type:
            resolution = int(res_val)
            break

    gpu_name = torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else "CPU"

    common_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_folder": model_folder, "model_name": model_name, "resolution": resolution,
        "world_size": 1, "gpu_name": gpu_name, "eval_batch_size": eval_batch_size,
        "num_simulations": num_simulations, "num_samples_per_sim": n_samples,
        "total_samples_generated": total_generated_samples, "total_time_s": round(total_time, 4),
        "total_generate_time_s": round(total_generate_time, 4), "io_overhead_time_s": round(io_overhead_time, 4),
        "generate_time_percent": round(generate_time_percent, 2), "peak_memory_gb": round(peak_memory_gb, 4),
        "sec_per_sample": round(sec_per_sample, 4),
    }

    file_exists = os.path.isfile(common_benchmark_file)
    with open(common_benchmark_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=common_fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(common_data)