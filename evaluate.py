# %%
import os
import numpy as np
import torch

# %%
root = 'Dataset'
model_folder = 'run/cosmos_dm_1900_2'
box_size = 1000
kmax = 0.4

# %%
import numpy as np
from tqdm import tqdm

import scipy.stats as stats
from typing import Dict, List, Tuple
import json
from metrics import ValidationSuite

# %%
val_suite = ValidationSuite(boxsize=box_size, kmax=kmax)

# %%
val_examples = []
    
# Dummy data for demonstration
for sample_no in range(1995, 2000):  # 100 validation examples
    sample_folder = os.path.join(model_folder, 'samples', str(sample_no))
    observation = np.load(os.path.join(sample_folder, 'observation.npy'))
    samples = np.load(os.path.join(sample_folder, 'sample.npy'))[:10]
    samples = samples.squeeze()
    truth = np.load(os.path.join(sample_folder, 'truth.npy'))

    # truth = (truth - np.mean(truth))/np.std(truth)
    val_examples.append((observation, samples, truth))

# Run evaluation
results = val_suite.evaluate_dataset(val_examples)

# Print summary
val_suite.print_summary(results)

# output_folder = './plots'
# if not os.path.exists(output_folder): 
#     os.makedirs(output_folder, exist_ok=True)
# val_suite.save_results(results, os.path.join(model_folder, 'results.csv'))