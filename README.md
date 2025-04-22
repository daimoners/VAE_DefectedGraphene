# GrapheNet: A Deep Learning Framework for Predicting the Physical and Electronic Properties of Nanographenes Using Images

This is the code used to produce the results presented in the paper called **GrapheNet: A Deep Learning Framework for Predicting the Physical and Electronic Properties of Nanographenes Using Images**.  
The paper is about using computer vision techniques (and in particular convolutional neural networks) to predict the electronic properties of graphene and graphene-oxyde flakes. The neural networks are implemented in `Python` using the `PyTorch` framework. Also the framewrok take care to create the dataset starting from a folder of `.xyz` files and a `.csv` file containing the target properties.

## Project Structure 
   ```
   ├── project/
   │ ├── config/ 
   │ │ ├── dataset.yaml  --> contain the config to create the dataset
   │ │ ├── train_predict_coulomb.yaml  --> contain the config to train on Coulomb eigenvalues
   │ │ ├── train_predict_coulomb.yaml  --> contain the config to train on png images with k-folds cross-validation
   │ │ └── train_predict.yaml    --> contain the config to train on png images
   │ ├── data/    --> contain the dataset and all the data (downloaded from the point 3 of the setup below)
   │ ├── lib/     --> contain all the necessary libraries for the dataset creation, models definition and training algorithms
   │ ├── outputs/ (folder created by hydra containing the experiments hystory)
   │ ├── environment.yml
   │ ├── kfold_train_lightning.py  --> script to train the model with k-folds cross-validation
   │ ├── dataset_generator.py  --> script to generate the png dataset from the xyz files
   │ ├── coulomb_predict_lightning.py    --> main script to evaluate the trained models on Coulomb eigenvalues
   │ ├── coulomb_train_lightning.py    --> main script to train the models on Coulomb eigenvalues
   │ ├── predict_lightning.py    --> main script to evaluate the trained models on png images
   │ └── train_lightning.py    --> main script to train the models on png images
   ```

## Usage

### Setup
1. Clone the repository and enter the `GrapheNet` directory:

   ```bash
   git clone -b published https://github.com/daimoners/GrapheNet.git  --depth 1 && cd GrapheNet
   ```

2. Create the conda env from the `environment.yaml` file and activate the conda env:

   ```bash
   conda env create -f environment.yaml && conda activate graphenet
   ```

3. Download the dataset and unzip it in the root folder:
   ```bash
   gdown 'https://drive.google.com/uc?id=1y7fKPrcQYfxmKkyzfhdOc0XJR1T88qYE' && unzip data_chapter_6.zip
   ```

### Configuration
1. In the data folder you can find the reference dataset of the paper, both for png images and as Coulomb matrices eigenvalues.
2. If you want to generate your own dataset, customize the `dataset.yaml` file in the `config` folder according to your needs, modifying only the following parameters:

   * `package_path`: specify the path of the root of the project.
   * `features`: specity a list of properties of interest, contained in the `original_dataset.csv`
   * `plot_distributions`: a flag that determines whether to plot the distributions of target properties for train/val/test.
   * `augmented_png`: a flag that determines whether to augment the images by rotating them by 90, 180, 270 degrees. (i.e. the size of the dataset will be quadrupled).
   * `augmented_xyz`: a flag that determines whether to augment the images by rotating them by a 3 randoms angle within [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]. (i.e. the size of the dataset will be quadrupled).

   The dataset can be created either randomly or by copying the same train/val/test samples used in another dataset (useful when testing different representations with the same dataset). If the field `from_csv.csv_dataset_path` is empty, the dataset is generated randomly.
   * `randomly`:
      ```
      n_items: total number of items in the dataset.
      train_split: sets the split percentage for the training set (i.e. 0.7 = 70%).
      val_split: sets the split percentage for the validation set.
      test_split: sets the split percentage for the test set.
      shuffle: a flag that determines whether to shuffle the dataset or not.
      oxygen_outliers_th: sets the threshold for the minimum average value (normalized on the dataset) of the distribution of oxygens in the flake.
      min_num_atoms: sets the minimum number of atoms for each sample in the dataset. 
      ```
   * `from_csv`:
      ```
      csv_dataset_path: path of the dataset from which you want to copy the train/val/test samples.
      ```

2. Customize your `train_predict.yaml` training configurations using Hydra. Configuration files are located in the `config` directory. Modify only the following parameters:

   * `package_path`: specify the path of the root of the project.
   * `atom_types`: specify the different type of atoms in the dataset (i.e. 3 for GO and 1 for DG).
   * `cluster`: a flag that determines whether the training should be performed on cluster or not. (if you use a cluster, modify accordingly the SLURM parameters at the bottom of the config file)
   * `coulomb`: a flag that determines whether the dataset is composed by coulomb matrices rather than png images.
   * `deterministic`: a flag that set the experiment to be reproducible.
   * `num_workers`: set the num workers for the Pytorch Dataloaders.
   * `cluster_num_workers`: set the num workers for the Pytorch Dataloaders when running on a cluster.
   * `enlargement_method`: you can choose between `padding` or `resize` and specify the enlargement method in order to have the same size for all the images (work only with images dataset).
   * `k_fold`: a flag that determines whether to run the training with 6 folds (cross-validation) or with the standard train/val/test split
   * `resolution`: sets the maximum size to which the images/coulomb matrices of the dataset will be resized.
   * `target`: sets the target property on which the network is trained.
   * `train`:
      ```
      base_lr: set the initial learning rate.
      batch_size: set the batch size.
      compile: a flag that set the model to be compiled or not.
      matmul_precision: set the pytorch matmul precision (high or medium).
      num_epochs: set the number of epochs.
      spath: sets the dataset on which the network will be trained.
      grayscale: a flag that set to convert the rgb images into grayscale for the training.
      network: specify the model to use for training/inference.
      ```

### Training with png

Launch the `train_lightning.py`:

   ```bash
   python train_lightning.py 
   ```
At the end of the training phase, the framework generates a yaml file containing some training parameters (batch_size, dataset, learning_rate, num_epochs, resolution, target, model_name) and the training time, along with the best checkpoints.

### Evaluation with png

The evaluation is automatically performed at the end of the training. If you want to perform again the evaluation on the test set, launch the `predict_lightning.py` with the appropriate hydra config file:

   ```bash
   python predict_lightning.py
   ```
At the end of the evaluation phase, the framework generates:
   * a png image containing the fit curve and the R2 value
   * the png images of the learning curves
   * a csv file containing for each sample of the test set, the predicted value, the real labelled value and the MAE errors.
   * a yaml file containing the MEAN, MAX and STD value of the MAE errors.


### (Optional) Training with Coulomb eigenvalues

Modify the desired target inside the `train_predict_coulomb.yaml` from the `config` folder according to your needs and launch the `coulomb_train_lightning.py`:

   ```bash
   python coulomb_train_lightning.py 
   ```
At the end of the training phase, the framework generates a yaml file containing some training parameters (batch_size, dataset, learning_rate, num_epochs, resolution, target, model_name) and the training time, along with the best checkpoints.

### (Optional) Evaluation with Coulomb eigenvalues

The evaluation is automatically performed at the end of the training. If you want to perform again the evaluation on the test set, launch the `coulomb_predict_lightning.py` with the appropriate hydra config file:

   ```bash
   python coulomb_predict_lightning.py
   ```
At the end of the evaluation phase, the framework generates:
   * a png image containing the fit curve and the R2 value
   * the png images of the learning curves
   * a csv file containing for each sample of the test set, the predicted value, the real labelled value and the MAE errors.
   * a yaml file containing the MEAN, MAX and STD value of the MAE errors.

### (Optional) Training with png images and k-folds cross-validation

Modify the desired target inside the `train_predict_kfold.yaml` from the `config` folder according to your needs and launch the `kfold_train_lightning.py`:

   ```bash
   python kfold_train_lightning.py
   ```
At the end of the training phase, the framework generates a yaml file containing some training parameters (batch_size, dataset, learning_rate, num_epochs, resolution, target, model_name) and the training time, along with the best checkpoints for each fold (default: 6 folds). It also provides a YAML file containing the MEAN, MAX, and STD values of the MAE errors averaged over the 6 folds.