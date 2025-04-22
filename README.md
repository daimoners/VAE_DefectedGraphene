# Extension to Synthetic Data Generation

This repository contains the implementation of a Convolutional Variational Autoencoder (VAE) trained to generate synthetic data of defective graphene nanoflakes with small, distributed defects.

## Project Structure 
   ```
   ├── project/
   │ ├── config/ 
   │ │ └── cfg.yaml  --> contain the config to train the VAE
   │ ├── data/    --> contain the dataset and all the data (downloaded from the point 3 of the setup below)
   │ ├── lib/     --> contain all the necessary libraries for the dataset creation, models definition and training algorithms
   │ ├── outputs/ (folder created by hydra containing the experiments hystory)
   │ ├── models/   --> the VAE checkpoints will be saved here
   │ ├── results/   --> the generated samples and the results will be saved here
   │ ├── environment.yml
   │ ├── dataset_generator.py  --> script to generate the dataset for the VAE training
   │ ├── features_analysis.ipynb    --> jupyter notebook to extract and compare the frequency features between the generated and original datasets 
   │ ├── generate_samples.py   --> main script to generate new samples from the trained VAE and discriminator models
   │ ├── train_discriminator.py     --> main script to train the discriminator
   │ └── train_vae.py    --> main script to train the VAE
   ```

## Usage

### Setup
1. Clone the repository and enter the `VAE_DefectedGraphene` directory:

   ```bash
   git clone -b published https://github.com/daimoners/VAE_DefectedGraphene.git  --depth 1 && cd VAE_DefectedGraphene
   ```

2. Create the conda env from the `environment.yaml` file and activate the conda env:

   ```bash
   conda env create -f environment.yaml && conda activate vae
   ```

3. Download the dataset and unzip it in the root folder:
   ```bash
   gdown 'https://drive.google.com/uc?id=1H_Yo8WfOPJaywNznRcU1FpKTNNacXU9Q' && unzip data_chapter_7_extension.zip
   ```

### Configuration
1. In the data folder you can find the reference dataset for the VAE training, along with the dataset and checkpoint of the discriminator model.
2. If you want to generate your own dataset, customize and launch the `dataset_generator.py` script.
3. Customize the `cfg.yaml` file in the `config` folder according to your needs, modifying only the following parameters:

   * `package_path`: specify the path of the root of the project.
   * `vae`:
      ```
      ch: base number of channels for the convolutional layers. Subsequent layers scale this number.
      blocks: scaling factors for the number of channels in each encoder/decoder stage.
      latent_channels: number of channels in the latent representation (z)
      deep_model: if True, enables a deeper architecture for encoder and decoder.
      ```
   * `train`:
      ```
      lr: set the initial learning rate.
      batch_size: set the batch size.
      image_size: set the input image size.
      nepoch: set the number of epochs.
      early_stop_patience: sets the early stop patience.
      scheduler_patience: sets the ReduceLROnPlateau stop patience.
      ```
    * (if you use a cluster, modify accordingly the SLURM parameters at the bottom of the config file)


### Train the VAE

Launch the `train_vae.py` script:

   ```bash
   python train_vae.py 
   ```
At the end of the training phase, the framework generates the best checkpoints in the `models/VAE` folder.

### (Optional) Train the discriminator

You can customize and launch the `train_discriminato.py` script to re-train the binary classification model:

   ```bash
   python train_discriminato.py
   ```
At the end of the training phase, the framework generates the best checkpoints in the `data/discriminator/model` folder and the confusion matrix in `data/discriminator/results`. The best checkpoints are already provided.

### Generate new samples

Customize and run the `generate_samples.py` script:
   ```bash
   python generate_samples.py
   ```
It will use the best models of the VAE and the discriminator, to generate 5000 samples (the number can be changed). The generated samples will be saved in `results/VAE/generated_dataset`.

### Comparison and analysis with the original dataset

You can run each block of the `features_analysis.ipynb` jupyter notebook to extract the frequency features from the generated synthetic dataset and compare the original and generated datasets in the frequency domain. (The dataframe of the original dataset and the generated synthetic dataset presented in the thesis are already provided).