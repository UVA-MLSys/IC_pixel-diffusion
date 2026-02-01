# %%
import os, sys
import numpy as np
from utils import get_config
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from utils import get_sigma_time, get_sample_time, VESDE, get_config
from model import UNet3DModel
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
import logging
import os
import sys
from os.path import join
import argparse

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
cosmo_dir = config.model.cosmo_dir
data_path = join(config.model.workdir, cosmo_dir)
checkpoint_dir = join(data_path, config.model.checkpoint_dir)

# %%
# Initialize score model
model = UNet3DModel(config)
#model = DataParallel(model)
model = model.to(DEVICE)

ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

sde = VESDE(
    config.model.sigma_min, config.model.sigma_max, 
    config.model.num_scales, 
    config.model.T, config.model.sampling_eps
)

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

# %%

for sample_no in range(args.start, args.end):
    print(f'Sampling {sample_no}')
    # z127_path = f"./Dataset/halo_LH_128/halo_lh_{sample_no}.npy" 
    z127_path = f"./Dataset/Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
    z0_path = f"./Dataset/Train_z0_2000/{sample_no}_z0.npy"

    # z127_path = f"../IC-Flow-Diffusion/Dataset/Train_z127_CAMELS/z127_{sample_no:04d}.npy"
    # z0_path = f"../IC-Flow-Diffusion/Dataset/Train_z0_CAMELS/z0_{sample_no:04d}.npy"

    output_dir = os.path.join(
        config.model.workdir, config.model.cosmo_dir,
        'samples', str(sample_no)
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

    # === Save as observation and truth ===
    np.save(os.path.join(output_dir, "observation.npy"), z0_noisy)
    np.save(os.path.join(output_dir, "truth.npy"), z127_norm)

    print(f"✅ Saved observation and truth to {output_dir}")


    # %%
    enable_tqdm = not args.disable_tqdm
    Nside = config.data.image_size
    #DEVICE = config.device

    sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
    sample_time = get_sample_time(config.model.sampling_eps, config.model.T)


    # %%
    input_data = np.float32(np.load(join(output_dir, 'observation.npy')))
    input_data.mean(), input_data.max(), input_data.min()

    # %%
    # Build pytorch dataloaders
    input_data = np.float32(np.load(join(output_dir, 'observation.npy')))
    print("Loaded shape:", input_data.shape)
    label_data = np.float32(np.load(join(output_dir, 'truth.npy')))
    input_data = torch.from_numpy(input_data).to(DEVICE)
    label_data = torch.from_numpy(label_data).to(DEVICE)
    input_data = torch.unsqueeze(input_data, dim=1)
    label_data = torch.unsqueeze(label_data, dim=1)


    # %%
    def one_step(x, t):
        t_vec = torch.ones(shape[0], device=DEVICE) * t
        model_output = model(torch.cat([x, input_data], dim=1), t_vec)
        x, x_mean = sde.update_fn(x, t_vec, model_output=model_output)
        return x, x_mean

    print("input_data shape before tiling:", input_data.shape)


    input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
    shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

    import time
    intermediate_dir = os.path.join(output_dir, 'intermediates')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir, exist_ok=True)

    samples = []
    print('Sampling begins.')
    for j in tqdm(
        range(config.sampling.num_samples//config.sampling.batch_size),
        disable=args.disable_tqdm
    ):
        with torch.no_grad(), ema.average_parameters():
            x = sde.prior_sampling(shape).to(DEVICE)
            timesteps = sde.timesteps.to(DEVICE)

            times = []
            start = time.perf_counter()

            for i in range(sde.N):
                t = timesteps[i]

                x, x_mean = one_step(x, t)
                if j==1: 
                    times.append(time.perf_counter()-start)

                filepath = os.path.join(intermediate_dir, f'{i}.npy')
                if j==1: np.save(filepath, x_mean.detach().cpu().numpy().squeeze())

            samples.append(x_mean.detach().cpu().numpy())

        if j==1: np.save(os.path.join(intermediate_dir, 'times.npy'), np.array(times))
        np.save(os.path.join(output_dir, 'sample.npy'), np.array(samples))
        print(f'Finished {j+1}th round')

    print('Done sampling')

    samples = np.array(samples).reshape(-1, Nside, Nside, Nside)
    np.save(os.path.join(output_dir, 'sample.npy'), samples)