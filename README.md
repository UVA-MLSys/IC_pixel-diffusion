# IC_pixel-diffusion: Pixel-based diffusion model for reconstructing cosmological initial conditions from Dark Matter Halos. 
This model reproduces the main architecture from the **"Posterior Sampling of the Initial Conditions of the Universe from Non-Linear Large Scale Structures Using Score-Based Generative Models"** ([arXiv:2304.03788](https://arxiv.org/abs/2304.03788)) and optimizes it for a new observation dataset. I trained it on **Quijote Latin Hypercube simulation dataset** to reconstruct the initial condition (z = 127) from DM halo (z=0).
## 1. Dataset Preparation

The dataset used for this project is based on the **Quijote simulation suite**, which provides
large-scale N-body simulations of the Universe. These simulations are used here to generate both
the **initial condition density fields (z = 127)** and the **halo density fields (z = 0)**.

The **initial condition (z = 127)** density fields are generated using the **Latin Hypercube simulation snapshots**
from Quijote. The corresponding generation script is provided here:
[Initial Condition Generation Code](<https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/Dataset/generate_train_z127_density.py>).

The **halo density fields (z = 0)** are constructed from the **halo catalogs** produced by the
**Friends-of-Friends (FoF)** algorithm applied to the Quijote N-body simulations.
The processing script used for this step is provided here:
[Halo Field Generation Code](<https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/Dataset/generate_halo_redshift_mass.py>).


After generating the individual samples for both redshifts (z = 127 and z = 0),
the **stacking script** in the `dataset/` folder is used to combine all simulation IDs
into two single large `.npy` arrays for training.
