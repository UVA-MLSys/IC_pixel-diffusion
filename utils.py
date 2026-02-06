import torch
import numpy as np
import json


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dict_to_config(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_config(v)
    return Config(**d)


def get_config(config_path):
    # Load parameters from a json file back into the Config class
    with open(config_path, 'r') as f:
        loaded_config_dict = json.load(f)

    # Convert dictionaries back into Config objects
    loaded_config = dict_to_config(loaded_config_dict)

    return loaded_config


# Define sigma(t) mapping
def get_sigma_time(sigma_min, sigma_max):
    def sigma_time(t):
        return sigma_min * (sigma_max / sigma_min) ** t
    return sigma_time

# Define time uniform sampling
def get_sample_time(sampling_eps, T):
    def sample_time(shape):
        return (sampling_eps - T) * torch.rand(shape) + T
    return sample_time



class VESDE():
  def __init__(self, sigma_min, sigma_max, N, T = 1, eps=1e-5):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N
    self.T = T
    self.eps = eps

    self.timesteps = torch.linspace(T, eps, N)

  def prior_sampling(self, shape):
    return torch.randn(*shape) * self.sigma_max

  def sample_time(self, shape):
    return (self.eps - self.T) * torch.rand(shape) + self.T

  def sigma_fn(self, t):
    return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

  def sde(self, x, t):
    sigma = self.sigma_fn(t)
    drift = torch.zeros_like(x)
    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                device=t.device))
    return drift, diffusion

  def rsde(self, x, t, model_output):
    """Create the drift and diffusion functions for the reverse SDE/ODE."""
    drift, diffusion = self.sde(x, t)
    score = self.score_fn(t, model_output)
    drift = drift - diffusion[:, None, None, None, None] ** 2 * score
    return drift, diffusion

  def score_fn(self, t, model_output):
    return model_output/self.sigma_fn(t)[:,None,None,None,None]

  def update_fn(self, x, t, model_output):
    dt = -self.T / self.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde(x, t, model_output)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


def get_filepath(sample_no, file_type):
    if file_type == 'z0':
        return f"Train_z0_2000/{sample_no}_z0.npy"
    elif file_type == 'z127':
        return f"Train_z127_from_IC_2000/df_m_z=127_sim{sample_no}.npy"
    elif file_type == 'halo':
        return f"halo_LH_128/halo_lh_{sample_no:04d}.npy"
    elif file_type == 'recon':
        return f"Recon_z127_2000/{sample_no}.npy"
    elif file_type == 'latent_z0':
        return f"Latent_z0/{sample_no}.npy"
    elif file_type == 'latent_z127':
        return f"Latent_z127/{sample_no}.npy"
    elif file_type == 'camels_z0':
        return f"Train_z0_CAMELS/z0_{sample_no:04d}.npy"
    elif file_type == 'camels_z127':
        return f"Train_z127_CAMELS/z127_{sample_no:04d}.npy"
    elif file_type == 'quijote_z0_32':
        return f'Quijote_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'quijote_z0_64':
        return f'Quijote_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'quijote_ic_32':
        return f'Quijote_processed/IC_32/{sample_no}.npy'
    elif file_type == 'quijote_ic_64':
        return f'Quijote_processed/IC_64/{sample_no}.npy'
    elif file_type == 'lc_ic_32':
        return f'latin_hypercube_LC_processed/IC_32/{sample_no}.npy'
    elif file_type == 'lc_ic_64':
        return f'latin_hypercube_LC_processed/IC_64/{sample_no}.npy'
    elif file_type == 'lc_ic_128':
        return f'latin_hypercube_LC_processed/IC_128/{sample_no}.npy'
    elif file_type == 'lc_z0_32':
        return f'latin_hypercube_LC_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'lc_z0_64':
        return f'latin_hypercube_LC_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'lc_z0_128':
        return f'latin_hypercube_LC_processed/Z0_128/{sample_no}.npy'
    elif file_type == 'bsq_ic_32':
        return f'BSQ_Processed/IC_32/{sample_no}.npy'
    elif file_type == 'bsq_ic_64':
        return f'BSQ_Processed/IC_64/{sample_no}.npy'
    elif file_type == 'bsq_ic_128':
        return f'BSQ_Processed/IC_128/{sample_no}.npy'
    elif file_type == 'bsq_z0_32':
        return f'BSQ_Processed/Z0_32/{sample_no}.npy'
    elif file_type == 'bsq_z0_64':
        return f'BSQ_Processed/Z0_64/{sample_no}.npy'
    elif file_type == 'bsq_z0_128':
        return f'BSQ_Processed/Z0_128/{sample_no}.npy'
    else:
        raise ValueError(f"Unknown file type: {file_type}")


stats_dict = {
    'z0': [-0.1235, 0.3096],
    'z127': [0, 0.00927],
    'halo': [0, 1.7225],
    'recon': [-0.1724, 0.356],
    'camels_z0': [-0.518061, 0.540111],
    'camels_z127': [0, 0.026746],
    'quijote_z0_32': [-0.0106, 0.0955],
    'quijote_z0_64': [-0.0335, 0.1681],
    'quijote_ic_32': [0.0000, 0.0033],
    'quijote_ic_64': [0.0000, 0.0057],
    'lc_z0_32': [0.9288, 0.1469],
    'lc_z0_64': [0.8878, 0.2294],
    'lc_z0_128': [0.8282, 0.2984],
    'lc_ic_32': [0.0000, 0.0029],
    'lc_ic_64': [0.0000, 0.0055],
    'lc_ic_128': [0.0000, 0.0091],
}