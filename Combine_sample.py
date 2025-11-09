import numpy as np

# Load the original sample
original_path = "/scratch/dye7jx/Projects/ICdiffusion/run/cosmo1999_dm_1900/sample0.npy"
samples = np.load(original_path)  # Shape: (25, 1, 1, 128, 128, 128)

# Reshape to remove singleton dimensions → (25, 128, 128, 128)
samples_reshaped = samples.reshape(-1, 128, 128, 128)

# Save as a new file (final version)
final_path = "/scratch/dye7jx/Projects/ICdiffusion/run/cosmo1999_dm_1900/sample.npy"
np.save(final_path, samples_reshaped)

print(f"✅ Final reshaped sample saved to: {final_path}")
print(f"✅ New shape: {samples_reshaped.shape}")
