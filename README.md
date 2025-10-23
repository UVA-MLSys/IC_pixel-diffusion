# IC_pixel-diffusion: Pixel-based diffusion model for reconstructing cosmological initial conditions from Dark Matter Halos. 
This model reproduces the main architecture from the **"Posterior Sampling of the Initial Conditions of the Universe from Non-Linear Large Scale Structures Using Score-Based Generative Models"** ([arXiv:2304.03788](https://arxiv.org/abs/2304.03788)) and optimizes it for a new observation dataset. I trained it on **Quijote Latin Hypercube simulation dataset** to reconstruct the initial condition (z = 127) from DM halo (z=0).

![halo_slice_z0](plots/input-target.png)


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

- [Full z = 0 dataset (Train_z0_2000.npy)](https://drive.google.com/drive/folders/1q6G-_9AL3xSll_kI4hf-qtSbotvebPuy?usp=drive_link)  
- [Full z = 127 dataset (Train_z127_2000.npy)](https://drive.google.com/drive/folders/1BO2AznTSw_31z-AjEa_gmPvNS8frHMcL?usp=drive_link)


</details>


<details>
<summary><b> 2. Model Training</b></summary>

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


<details>
<summary><b>3. Sampling and Evaluation</b></summary>

After training, the model enters the **sampling phase**, where it generates reconstructed
initial conditions from unseen test data. During sampling, the model receives the **observed z = 0 halo field**
as input and progressively denoises it to reconstruct the corresponding **z = 127 initial condition field**.

The sampling process is handled by the following script:
[`sample.py`](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/sample.py)

The **number of generated samples** can be adjusted as a hyperparameter in the configuration file, allowing flexibility in testing on different dataset sizes.

Once the samples are generated, they are combined into a single file using the stacking script:
[`combine_samples.py`](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/Combine_sample.py)

This combined sample file is then used to evaluate the model’s reconstruction performance.
The evaluation is performed using:
[`result.py`](https://github.com/UVA-MLSys/IC_pixel-diffusion/blob/main/results.py)

The evaluation script computes three key metrics to assess reconstruction quality:
- **Power Spectrum** — measures the statistical similarity of large-scale modes.  
- **Cross-Correlation Coefficient** — quantifies the correlation between predicted and true fields.  
- **Transfer Function** — evaluates the scale-dependent amplitude accuracy.


The figure below shows the evaluation results for the model trained on **1900 samples** and
**conditioned on the halo density field**.  
It presents the three key metrics—**Power Spectrum**, **Cross-Correlation Coefficient**, and
**Transfer Function**—used to assess the reconstruction performance of the model.

<p align="center">
  <img src="plots/eval_plot.png"
       alt="Evaluation metrics"
       width="480">
</p>




</details>

