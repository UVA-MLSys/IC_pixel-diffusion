# %%
import os, sys
import numpy as np
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
    
    return parser

parser = get_parser()
args = parser.parse_args()

# %%
config =get_config(args.config)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

model_folder = join(config.model.workdir, config.model.cosmo_dir)
checkpoint_dir = join(model_folder, config.model.checkpoint_dir)
data_root = config.data.path

input_type = config.data.input_type
target_type = config.data.target_type

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

model.eval()

for sample_no in tqdm(range(args.start, args.end), disable=args.disable_tqdm):
    z127_path = os.path.join(data_root, get_filepath(sample_no, target_type)) #  f"./Dataset/Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
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

    # === Load z=127 and normalize ===
    z127 = np.load(z127_path).reshape(N, N, N)
    z127_norm = (z127 - np.mean(z127)) / np.std(z127)
    z127_norm = z127_norm[np.newaxis, ...]

    np.save(os.path.join(output_dir, "truth.npy"), z127_norm)

    Nside = config.data.image_size
    #DEVICE = config.device

    sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
    sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

    label_data = np.float32(np.load(join(output_dir, 'truth.npy')))
    input_data = torch.from_numpy(np.float32(z0_noisy)).to(DEVICE)
    label_data = torch.from_numpy(label_data).to(DEVICE)
    input_data = torch.unsqueeze(input_data, dim=1)
    label_data = torch.unsqueeze(label_data, dim=1)

    input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
    shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

    timesteps = sde_ddim.get_ddim_schedule(inference_steps + 1, method=schedule_method)

    samples = []
    for j in range(config.sampling.num_samples // config.sampling.batch_size):
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
            samples.append(x.detach().cpu().numpy())
        
        # Save intermediate results
        np.save(join(output_dir, 'sample.npy'), np.array(samples))
        # print(f'Finished batch {j+1}')

    samples = np.array(samples).reshape(-1, Nside, Nside, Nside)
    np.save(os.path.join(output_dir, 'sample.npy'), samples)