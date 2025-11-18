import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import get_sigma_time, get_sample_time, get_config
from model import UNet3DModel
torch.backends.cudnn.benchmark = True
import os
import logging
from torch_ema import ExponentialMovingAverage
import torch.amp
import argparse

# --- DDP SETUP FUNCTION ---
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

# --- Main Script ---
parser = get_parser()
args = parser.parse_args()
config = get_config(args.config)
enable_ddp = not args.disable_ddp

local_rank = setup_ddp() if enable_ddp else 0
DEVICE = torch.device(f'cuda:{local_rank}')
is_main_process = local_rank == 0

output_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

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

sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

scaler = torch.amp.GradScaler("cuda")

def train_one_epoch(
    training_loader, model, optimizer, 
    ema, scaler, epoch, scheduler
):
    model.train()
    if enable_ddp: training_loader.sampler.set_epoch(epoch)
    avg_loss = 0.
    counter = 0
    progress_bar = tqdm(
        training_loader, desc=f"Training Epoch {epoch+1}", 
        disable=not is_main_process or args.disable_tqdm
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
        
        avg_loss += loss.item()

        counter += 1
        progress_bar.set_postfix({'loss': f'{avg_loss/counter:.4g}'})

    scheduler.step(avg_loss / counter)
        
    return avg_loss / counter

if is_main_process:
    logging.info("💾 Loading data on all processes...")

input_data = np.float32(np.load(config.data.path + 'quijote128_halo_train_100.npy')) # at z0 # originally quijote128_z0_train_1900
label_data = np.float32(np.load(config.data.path + 'quijote128_z127_train_100.npy')) # at z inf or 12.7 here # originally quijote128_z127_train_1900

# normalize
# input_data = (input_data - np.mean(input_data, axis=(1, 2, 3), keepdims=True)) / np.std(input_data, axis=(1, 2, 3), keepdims=True)
label_data = (label_data - np.mean(label_data, axis=(1, 2, 3), keepdims=True)) / np.std(label_data, axis=(1, 2, 3), keepdims=True)

input_data = torch.from_numpy(input_data)
label_data = torch.from_numpy(label_data)
input_data = torch.unsqueeze(input_data, dim=1)
label_data = torch.unsqueeze(label_data, dim=1)
train_dataset = TensorDataset(input_data, label_data)

if is_main_process:
    logging.info("✅ Data loaded.")
    
if enable_ddp:
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    training_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
else:
    training_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

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
init_epoch = 0

if is_main_process:
    logging.info("🔁 Starting training loop.")
import csv 
loss_file_name = os.path.join(output_dir, "Train.csv")
with open(loss_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss'])

# %%
for epoch in range(config.training.n_epochs):
    loss = train_one_epoch(
        training_loader, model, optimizer, ema, 
        scaler, epoch, scheduler
    )
    if is_main_process:
        logging.info(f"Epoch {epoch+1}/{config.training.n_epochs} - Loss: {loss:.6g}")
        state_dict = model.module.state_dict() if enable_ddp else model.state_dict()

        torch.save(
            dict(
                optimizer=optimizer.state_dict(), model=state_dict, 
                ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch
            ),
            os.path.join(checkpoint_dir, 'checkpoint.pth')
        )
        if epoch % 10 == 0:
            torch.save(
                dict(
                    optimizer=optimizer.state_dict(), model=state_dict, 
                    ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch
                ),
                os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
            )
            
        with open(loss_file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, loss])

if is_main_process:
    logging.info("🎉 Training complete.")

cleanup_ddp()
