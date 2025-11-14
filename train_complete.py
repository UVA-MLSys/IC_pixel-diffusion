# %% [markdown]
# # Imports

# %%
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import get_sigma_time, get_sample_time, get_config
from model import UNet3DModel
# torch.backends.cudnn.benchmark = True
import os
import logging
from torch_ema import ExponentialMovingAverage
import argparse
from torch.utils.data import Dataset
from pathlib import Path

# %% [markdown]
# # Initialization

# %%
def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def get_parser():
    parser = argparse.ArgumentParser(
        description='Run Diffusion Model', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--disable_tqdm', action='store_true', help='whether to enable tqdm progress bar')
    parser.add_argument('--disable_ddp', action='store_true', help='whether to enable distributed data parallel')
    parser.add_argument('--num_workers', type=int, default=3, help='number of workers for dataloader')
    
    return parser

# %%

parser = get_parser()
args = parser.parse_args()
config = get_config(args.config)
enable_tqdm = not args.disable_tqdm
enable_ddp = not args.disable_ddp

local_rank = setup_ddp() if enable_ddp else 0
DEVICE = torch.device(f'cuda:{local_rank}')
is_main_process = local_rank == 0
output_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# %%
if is_main_process:
    if enable_ddp:
        print("🚀 Using DistributedDataParallel (DDP) for training.")
        print("🔍 Number of GPUs being used:", dist.get_world_size())
    checkpoint_dir = os.path.join(output_dir, config.model.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    gfile_stream = open(os.path.join(output_dir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

# %% [markdown]
# # Dataset

# %%


def list_files(directory, ext='h5'):
    return [str(file) for file in Path(directory).rglob(f"*.{ext}")]

def get_filepath(sample_no, file_type):
    if file_type == 'z0':
        return f"Train_z0_2000/{sample_no}_z0.npy"
    elif file_type == 'z127':
        return f"Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
    elif file_type == 'halo':
        return f"halo_LH_128/halo_lh_{sample_no:04d}.npy"
    else:
        raise ValueError(f"Unknown file type: {file_type}")

class SimulationDataset(Dataset):
    def __init__(
        self, root, n_samples=None,
        input_type='z0', target_type='z127'
    ):
        self.root = root
        
        self.input_files = [
            os.path.join(root, get_filepath(sample_no, input_type))
            for sample_no in range(n_samples)
        ]
        self.target_files = [
            os.path.join(root, get_filepath(sample_no, target_type))
            for sample_no in range(n_samples)
        ]
        
        # sanity check
        for input_file, target_file in zip(self.input_files, self.target_files):
            assert os.path.exists(input_file), f'{input_file} does not exist'
            assert os.path.exists(target_file), f'{target_file} does not exist'
        
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        input = np.load(self.input_files[idx])
        input = torch.from_numpy(input).unsqueeze(0)
        
        label  = np.load(self.target_files[idx])
        label = torch.from_numpy(label).unsqueeze(0)
        return input, label

# %% [markdown]
# # Training Utils

# %%
sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)
scaler = torch.amp.GradScaler("cuda")

# %%
def train_one_epoch(training_loader, model, optimizer, ema, scaler, epoch, scheduler):
    model.train()
    if enable_ddp: training_loader.sampler.set_epoch(epoch)
    avg_loss = 0.
    counter = 0
    progress_bar = tqdm(
        training_loader, desc=f"Training Epoch {epoch+1}", 
        disable=not (is_main_process and enable_tqdm)
    )
    
    for i, data_list in enumerate(progress_bar):
        input_data = data_list[0].to(DEVICE, non_blocking=True)
        label_data = data_list[1].to(DEVICE, non_blocking=True)
        B = label_data.size(dim=0)
        input_data += config.data.noise_sigma * torch.randn_like(input_data)
        
        time_steps = sample_time(shape=(B,)).to(DEVICE)
        sigmas = sigma_time(time_steps).to(DEVICE)
        sigmas = sigmas[:, None, None, None, None]
        z = torch.randn_like(label_data)
        inputs = torch.cat([label_data + sigmas * z, input_data], dim=1)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast("cuda"):
            output = model(inputs, time_steps)
            loss = torch.sum(torch.square(output + z)) / B
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        ema.update()
        scheduler.step(loss)
        avg_loss += loss.item()

        progress_bar.set_postfix({'loss': f'{avg_loss:.4g}'})
        counter += 1
    return avg_loss / counter

# %%
if is_main_process:
    logging.info("Loading data on all processes...")

# %% [markdown]
# # Dataloader

# %%
# config.data.sample_size = 100
dataset = SimulationDataset(
    config.data.path,
    n_samples=config.data.sample_size,
    input_type=config.data.input_type,
    target_type=config.data.target_type
)

# %%
if enable_ddp:
    train_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    training_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
else:
    training_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

# %% [markdown]
# # UNet3DModel

# %% [markdown]
# ## Class

# %% [markdown]
# ## Initialize

# %%
model = UNet3DModel(config).to(DEVICE)

if enable_ddp:
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.optim.lr,
    betas=(config.optim.beta1, 0.999),
    eps=config.optim.eps,
    weight_decay=config.optim.weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, min_lr=1e-7
)

ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

if is_main_process:
    logging.info("Starting training loop.")

# %% [markdown]
# ## Train

import csv 
loss_file_name = os.path.join(output_dir, "Train.csv")
with open(loss_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss'])

# %%
for epoch in range(config.training.n_epochs):
    loss = train_one_epoch(training_loader, model, optimizer, ema, scaler, epoch, scheduler)
    if is_main_process:
        logging.info(f"Epoch {epoch+1}/{config.training.n_epochs} - Loss: {loss:.6g}")
        state_dict = model.module.state_dict() if enable_ddp else model.state_dict()

        torch.save(
            dict(optimizer=optimizer.state_dict(), model=state_dict, ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch),
            os.path.join(checkpoint_dir, 'checkpoint.pth')
        )
        if epoch % 10 == 0:
            torch.save(
                dict(optimizer=optimizer.state_dict(), model=state_dict, ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch),
                os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
            )
            
        with open(loss_file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, loss])

# %% [markdown]
# # Make Observation

# %%
root = config.data.path
test_sample_no = 0
z127_path =  os.path.join(root, get_filepath(test_sample_no, file_type=config.data.input_type))
z0_path = os.path.join(root, get_filepath(test_sample_no, file_type=config.data.target_type))

# %%
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

# %% [markdown]
# # Sample

# %% [markdown]
# ## Observation and Truth

# %%
from os.path import join

Nside = config.data.image_size

input_data = np.float32(np.load(join(output_dir, 'observation.npy')))
print("Loaded shape:", input_data.shape)
label_data = np.float32(np.load(join(output_dir, 'truth.npy')))
input_data = torch.from_numpy(input_data).to(DEVICE)
label_data = torch.from_numpy(label_data).to(DEVICE)
input_data = torch.unsqueeze(input_data, dim=1)
label_data = torch.unsqueeze(label_data, dim=1)

# %% [markdown]
# ## Initialize

# %%
from utils import VESDE

# Initialize score model
model = UNet3DModel(config)
#model = DataParallel(model)
model = model.to(DEVICE)

# Define optimizer
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay                   
        )
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

sde = VESDE(config.model.sigma_min, config.model.sigma_max, config.model.num_scales, config.model.T, config.model.sampling_eps)

# %%
# Check for existing checkpoint
checkpoint_path = join(checkpoint_dir, 'checkpoint.pth')
if os.path.isfile(checkpoint_path):
    loaded_state = torch.load(checkpoint_path, map_location=DEVICE)
    optimizer.load_state_dict(loaded_state['optimizer'])
    model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])
    init_epoch = int(loaded_state['epoch'])
    logging.warning(f"Loaded checkpoint from {checkpoint_path}.")
else:
    logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")


# %%
shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

def one_step(x, t):
    t_vec = torch.ones(shape[0], device=DEVICE) * t
    model_output = model(torch.cat([x, input_data], dim=1), t_vec)
    x, x_mean = sde.update_fn(x, t_vec, model_output=model_output)
    return x, x_mean

# %%
model.eval()
input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
task_id = 0

samples = []
print('Sampling begins.')
final_path = os.path.join(output_dir, f'sample{task_id}.npy')
for j in tqdm(
    range(config.sampling.num_samples//config.sampling.batch_size),
    disable=not (is_main_process and enable_tqdm)
):
    with torch.no_grad(), ema.average_parameters():
        x = sde.prior_sampling(shape).to(DEVICE)
        timesteps = sde.timesteps.to(DEVICE)
        for i in range(sde.N):
            t = timesteps[i]
            x, x_mean = one_step(x, t)
        samples.append(x_mean.detach().cpu().numpy())
    np.save(final_path, np.array(samples))
    
np.save(final_path, np.array(samples))
print(f"Sample saved to: {final_path}")
