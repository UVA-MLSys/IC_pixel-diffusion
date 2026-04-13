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

class VESDE_DDIM(VESDE):
    def __init__(self, sigma_min, sigma_max, N, T=1, eps=1e-5):
        super().__init__(sigma_min, sigma_max, N, T, eps)

    def get_ddim_schedule(self, num_inference_steps, method="quadratic"):
        """
        Generates time steps for the sampling process.
        Adapts the logic from the Keras simulation_ddim function.
        """
        if method == "linear":
            # Standard linear interpolation in time
            t_steps = torch.linspace(self.T, self.eps, num_inference_steps)
            
        elif method == "quadratic":
            # Quadratic interpolation in sqrt-time space
            # This matches: np.linspace(0, np.sqrt(T), n) ** 2
            # It concentrates steps near t=eps (where fine details form)
            
            # We interpolate linearly between sqrt(T) and sqrt(eps)
            sqrt_T = np.sqrt(self.T)
            sqrt_eps = np.sqrt(self.eps)
            
            # Create linear grid in sqrt space
            lin_grid = torch.linspace(sqrt_T, sqrt_eps, num_inference_steps)
            
            # Square it back to get the quadratic time schedule
            t_steps = lin_grid ** 2
            
        return t_steps

    def ddim_step(self, x, t, t_prev, model_output, eta=0.0):
        """
        Deterministic DDIM update for Variance Exploding SDE.
        
        Args:
            x: Current state x_t
            t: Current time
            t_prev: Next time step (closer to 0)
            model_output: The raw output from the neural network
            eta: 0.0 for deterministic (ODE), >0 for stochasticity
        """
        # 1. Get noise scales (sigmas) for current and next step
        sigma_t = self.sigma_fn(t)[:, None, None, None, None]
        sigma_prev = self.sigma_fn(t_prev)[:, None, None, None, None]
        
        # 2. Extract predicted noise (epsilon)
        # In VE-SDE: Score = model_output / sigma_t
        # And Score ~= -epsilon / sigma_t
        # Therefore: model_output ~= -epsilon
        # So: epsilon_pred = -model_output
        eps_pred = -model_output

        # 3. Predict x_0 (clean data)
        # x_0 = x_t - sigma_t * epsilon
        x_0_pred = x - sigma_t * eps_pred
        
        # 4. Compute DDIM Variance (usually 0 for deterministic sampling)
        # This allows for interpolation between ODE (eta=0) and SDE (eta=1)
        sigma_tau = eta * torch.sqrt(
            (sigma_prev**2 / sigma_t**2) * (1 - (sigma_prev**2 / sigma_t**2)) # Simplified term for VE
        )
        # Note: For pure VE, strict DDIM sigma calculation is slightly different 
        # but for eta=0 (which is the goal of DDIM), the noise term vanishes anyway.
        
        # 5. Compute the direction to x_{t_prev}
        # Direction = sqrt(sigma_prev^2 - sigma_tau^2) * epsilon
        dir_xt = torch.sqrt(sigma_prev**2 - sigma_tau**2) * eps_pred
        
        # 6. Random noise (if eta > 0)
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        
        # 7. Final Update
        x_prev = x_0_pred + dir_xt + sigma_tau * noise
        
        return x_prev, x_0_pred


def get_filepath(sample_no, file_type):
    if file_type == 'z0':
        return f"Quijote_processed/Z0_128/{sample_no}_z0.npy"
    elif file_type == 'z127':
        return f"Quijote_processed/IC_128/df_m_z=127_sim{sample_no}.npy"
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
    elif file_type == 'eq_ic_32':
        return f'latin_hypercube_EQ_processed/IC_32/{sample_no}.npy'
    elif file_type == 'eq_ic_64':
        return f'latin_hypercube_EQ_processed/IC_64/{sample_no}.npy'
    elif file_type == 'eq_ic_128':
        return f'latin_hypercube_EQ_processed/IC_128/{sample_no}.npy'
    elif file_type == 'eq_z0_32':
        return f'latin_hypercube_EQ_processed/Z0_32/{sample_no}.npy'
    elif file_type == 'eq_z0_64':
        return f'latin_hypercube_EQ_processed/Z0_64/{sample_no}.npy'
    elif file_type == 'eq_z0_128':
        return f'latin_hypercube_EQ_processed/Z0_128/{sample_no}.npy'
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
    'lc_z0_32': [-0.0093, 0.0895],
    'lc_z0_64': [-0.0334, 0.1683],
    'lc_z0_128': [-0.0891, 0.2674],
    'lc_ic_32': [0.0000, 0.0029],
    'lc_ic_64': [0.0000, 0.0055],
    'lc_ic_128': [0.0000, 0.0091],

    # bsq dataset
    'bsq_z0_32': [-0.0108, 0.0963],
    'bsq_z0_64': [-0.0338, 0.1688],
    'bsq_z0_128': [-0.0854, 0.2621],
    'bsq_ic_32': [0.0000, 0.0033],
    'bsq_ic_64': [0.0000, 0.0057],
    'bsq_ic_128': [-0.0000, 0.0092],

    # eq dataset
    'eq_z0_32': [0.9287, 0.1465],
    'eq_z0_64': [0.8908, 0.2239],
    'eq_z0_128': [0.8352, 0.2910],
    'eq_ic_32': [0.0000, 0.0033],
    'eq_ic_64': [0.0000, 0.0057],
    'eq_ic_128': [0.0000, 0.0093]
}