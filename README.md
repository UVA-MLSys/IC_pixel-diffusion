# IC_pixel-diffusion: Pixel-based diffusion model for reconstructing cosmological initial conditions from Dark Matter Halos. 
This model reproduces the main architecture from the **"Posterior Sampling of the Initial Conditions of the Universe from Non-Linear Large Scale Structures Using Score-Based Generative Models"** ([arXiv:2304.03788](https://arxiv.org/abs/2304.03788)) and optimizes it for a new observation dataset. I trained it on **Quijote Latin Hypercube simulation dataset** to reconstruct the initial condition (z = 127) from DM halo (z=0).
<details>
<summary><b> 1. Dataset Preparation</b></summary>

The dataset used for this project is based on the **Quijote simulation suite**, which provides
large-scale N-body simulations of the Universe. These simulations are used here to generate both
the **initial condition density fields (z = 127)** and the **halo density fields (z = 0)**.

The **initial condition (z = 127)** density fields are generated using the **Latin Hypercube simulation snapshots**
from Quijote. The corresponding generation script is provided here:
[Initial Condition Generation Code](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/Dataset/generate_train_z127_density.py).

The **halo density fields (z = 0)** are constructed from the **halo catalogs** produced by the
**Friends-of-Friends (FoF)** algorithm applied to the Quijote N-body simulations.
The processing script used for this step is provided here:
[Halo Field Generation Code](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/Dataset/generate_halo_redshift_mass.py).

After generating the individual samples for both redshifts (z = 127 and z = 0),
the **stacking script** in the `dataset/` folder is used to combine all simulation IDs
into two single large `.npy` arrays for training.


For demonstration purposes, two small **stacked dataset samples** are included in the `Dataset/` folder:

- `quijote128_halo_train_3.npy` — stacked sample of the **z = 0** halo density fields (3 simulations)  
- `quijote128_z127_train_3.npy` — stacked sample of the **z = 127** initial condition fields (3 simulations)

These example files allow users to verify the dataset format and test the training and sampling scripts without downloading the full dataset.

The **complete dataset** (2000 generated samples for each redshift) is available for download from Google Drive:

- [Full z = 0 dataset (Train_z0_2000.npy)](<add-your-google-drive-link-here>)  
- [Full z = 127 dataset (Train_z127_2000.npy)](<add-your-google-drive-link-here>)


</details>


<details>
<summary><b> Model Training</b></summary>

The stacked datasets of both redshifts (**z = 0** halo fields and **z = 127** initial condition fields)
are fed into the conditional diffusion model to begin training.  
A total of **1900 samples** are used for training for each redshift. 
The corresponding training script is provided here: [train code](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/train.py)
Training is performed on **4 NVIDIA A100 GPUs** available on the **UVA Rivanna** supercomputing cluster,
using a **batch size of 4 per GPU** (effective total batch size of 16) for **400 epochs**.  
The complete training process takes approximately **17 hours**.

All key hyperparameters, such as the number of epochs, batch size, learning rate, and model configuration,
can be modified in the corresponding [config file](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/config.json) to suit different datasets or experiments.

</details>


