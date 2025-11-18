import numpy as np
from utils import get_config
import os, sys


config_filename = str(sys.argv[1]) 
config = get_config(config_filename)
task_ids = [0]
Nside = config.data.image_size

input_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)

# Load the original sample
samples = []
for task_id in task_ids:
    original_path = os.path.join(input_dir, f'sample{task_id}.npy')
    sample = np.load(original_path)  # Shape: (25, 1, 1, 128, 128, 128)
    samples.append(sample)

# Reshape to remove singleton dimensions → (25, 128, 128, 128)
samples_reshaped = np.array(samples).reshape(-1, Nside, Nside, Nside)

# Save as a new file (final version)
final_path = os.path.join(input_dir, f'sample.npy')
np.save(final_path, samples_reshaped)

print(f"✅ Final reshaped sample saved to: {final_path}")
print(f"✅ New shape: {samples_reshaped.shape}")
