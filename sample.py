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
    parser.add_argument('--task_id', type=int, default=0, help='Task id for sample')
    parser.add_argument('--config', type=str, default='config.json', help='Configuration file')
    parser.add_argument(
        '--disable_tqdm', action='store_true', 
        help='whether to enable tqdm progress bar'
    )
    
    return parser

parser = get_parser()
args = parser.parse_args()
config = get_config(args.config)

task_id = args.task_id 
config_filename = args.config
enable_tqdm = not args.disable_tqdm
config = get_config(config_filename)


Nside = config.data.image_size
#DEVICE = config.device
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

cosmo_dir = config.model.cosmo_dir
data_path = join(config.model.workdir, cosmo_dir)
checkpoint_dir = join(data_path, config.model.checkpoint_dir)

# Build pytorch dataloaders
input_data = np.float32(np.load(join(data_path, 'observation.npy')))
print("Loaded shape:", input_data.shape)
label_data = np.float32(np.load(join(data_path, 'truth.npy')))
input_data = torch.from_numpy(input_data).to(DEVICE)
label_data = torch.from_numpy(label_data).to(DEVICE)
input_data = torch.unsqueeze(input_data, dim=1)
label_data = torch.unsqueeze(label_data, dim=1)

# Initialize score model
model = UNet3DModel(config)
#model = DataParallel(model)
model = model.to(DEVICE)
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

sde = VESDE(config.model.sigma_min, config.model.sigma_max, config.model.num_scales, config.model.T, config.model.sampling_eps)

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

def one_step(x, t):
    t_vec = torch.ones(shape[0], device=DEVICE) * t
    model_output = model(torch.cat([x, input_data], dim=1), t_vec)
    x, x_mean = sde.update_fn(x, t_vec, model_output=model_output)
    return x, x_mean

print("input_data shape before tiling:", input_data.shape)


input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

samples = []
print('Sampling begins.')
for j in tqdm(
    range(config.sampling.num_samples//config.sampling.batch_size),
    disable=args.disable_tqdm
):
    with torch.no_grad(), ema.average_parameters():
        x = sde.prior_sampling(shape).to(DEVICE)
        timesteps = sde.timesteps.to(DEVICE)
        for i in tqdm(range(sde.N), disable=args.disable_tqdm):
            t = timesteps[i]

            start = time.perf_counter()
            x, x_mean = one_step(x, t)
            times.append(time.perf_counter()-start)

            filepath = os.path.join(output_dir, f'{i}.npy')
            np.save(filepath, x_mean.detach().cpu().numpy().squeeze())

        samples.append(x_mean.detach().cpu().numpy())
    np.save(data_path + 'sample{}.npy'.format(task_id), np.array(samples))
    print(f'Finished {j+1}th round')

print('Done sampling')
np.save(data_path + 'sample{}.npy'.format(task_id), np.array(samples))
