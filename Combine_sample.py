import numpy as np
from utils import get_config
import os

config = get_config('./config.json')
task_id = 0

input_dir = os.path.join(config.model.workdir, config.model.cosmo_dir)

# Load the original sample
original_path = os.path.join(input_dir, f'sample{task_id}.npy')
samples = np.load(original_path)  # Shape: (25, 1, 1, 128, 128, 128)

# Reshape to remove singleton dimensions → (25, 128, 128, 128)
samples_reshaped = samples.reshape(-1, 128, 128, 128)

# Save as a new file (final version)
final_path = os.path.join(input_dir, f'sample.npy')
np.save(final_path, samples_reshaped)

print(f"✅ Final reshaped sample saved to: {final_path}")
print(f"✅ New shape: {samples_reshaped.shape}")
