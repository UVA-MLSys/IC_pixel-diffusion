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

# --- Main Script ---
local_rank = setup_ddp()
DEVICE = torch.device(f'cuda:{local_rank}')
is_main_process = local_rank == 0

config = get_config('./config.json')

if is_main_process:
    print("🚀 Using DistributedDataParallel (DDP) for training.")
    print("🔍 Number of GPUs being used:", dist.get_world_size())
    checkpoint_dir = os.path.join(config.model.workdir, "checkpoints_2400_halos+dm_128_ddp")
    os.makedirs(checkpoint_dir, exist_ok=True)

    gfile_stream = open(os.path.join(config.model.workdir, 'stdout_2400_halos+dm_128_ddp.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')

sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

scaler = torch.amp.GradScaler("cuda")

def train_one_epoch(training_loader, model, optimizer, ema, scaler, epoch):
    model.train()
    training_loader.sampler.set_epoch(epoch)
    avg_loss = 0.
    counter = 0
    progress_bar = tqdm(training_loader, desc=f"Training Epoch {epoch+1}", disable=not is_main_process)
    
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
    return avg_loss / counter

if is_main_process:
    logging.info("💾 Loading data on all processes...")

input_data = np.float32(np.load(config.data.path + 'quijote128_z0+halo_train_2400.npy'))
label_data = np.float32(np.load(config.data.path + 'quijote128_z127_train_2400.npy'))
label_data = (label_data - np.mean(label_data, axis=(1, 2, 3), keepdims=True)) / np.std(label_data, axis=(1, 2, 3), keepdims=True)
input_data = torch.from_numpy(input_data)
label_data = torch.from_numpy(label_data)
input_data = torch.unsqueeze(input_data, dim=1)
label_data = torch.unsqueeze(label_data, dim=1)
train_dataset = TensorDataset(input_data, label_data)

if is_main_process:
    logging.info("✅ Data loaded.")
    
train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
training_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

model = UNet3DModel(config).to(DEVICE)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.optim.lr,
    betas=(config.optim.beta1, 0.999),
    eps=config.optim.eps,
    weight_decay=config.optim.weight_decay
)
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
init_epoch = 0

if is_main_process:
    logging.info("🔁 Starting training loop.")

for epoch in range(init_epoch, config.training.n_epochs + 1):
    loss = train_one_epoch(training_loader, model, optimizer, ema, scaler, epoch)
    if is_main_process:
        logging.info(f"Epoch {epoch+1}/{config.training.n_epochs} - Loss: {loss:.6f}")
        torch.save(
            dict(optimizer=optimizer.state_dict(), model=model.module.state_dict(), ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch),
            os.path.join(config.model.workdir, "checkpoints_2400_halos+dm_128_ddp", 'checkpoint.pth')
        )
        if epoch % 10 == 0:
            torch.save(
                dict(optimizer=optimizer.state_dict(), model=model.module.state_dict(), ema=ema.state_dict(), scaler=scaler.state_dict(), epoch=epoch),
                os.path.join(config.model.workdir, "checkpoints_2400_halos+dm_128_ddp", f'checkpoint_{epoch}.pth')
            )

if is_main_process:
    logging.info("🎉 Training complete.")

cleanup_ddp()
