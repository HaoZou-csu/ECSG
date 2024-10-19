## ECSG

Machine learning offers a promising avenue for expediting the discovery of new compounds by accurately predicting their thermodynamic stability. This approach provides significant advantages in terms of time and resource efficiency compared to traditional experimental and modeling methods. However, most existing models are constructed based on specific domain knowledge, potentially introducing biases that impact their performance. 

To overcome this limitation, we propose a novel machine learning framework **Electron Configuration Stacked Generalization (ECSG)**, rooted in electron configuration, further enhanced through stack generalization with two additional models grounded in diverse domain knowledge. Experimental results validate the efficacy of our model in accurately predicting the stability of compounds, achieving an impressive Area Under the Curve (AUC) score of 0.988. Notably, our model demonstrates exceptional efficiency in sample utilization, requiring only one-seventh of the data used by existing models to achieve the same performance. 
## Contents
- [Installation](#installation)
- [Demo data](#demo-data)
- [Input](#input)
- [Output](#output)
- [Usage](#usage)
- [Experiment reproduction](#experiment-reproduction)
- [Contact](#contact)

## Installation

#### Required packeages

To use this project, you need the following packages installed:

[Python3.*](https://www.python.org/) (version>=3.8)\
[PyTorch](https://pytorch.org/) (version >=1.9.0, <=1.16.0) \
[matminer](https://hackingmaterials.lbl.gov/matminer/)\
[pymatgen](https://pymatgen.org/)\
numpy\
pandas\
scikit-learn\
torch_geometric\
torch-scatter\
tqdm\
xgboost\
scipy\
pytest\
smact

If you encounter issues installing `torch-scatter` or experience errors related to `torch-scatter` during runtime, please uninstall `torch-scatter`. After uninstallation, custom functions based on PyTorch will be called instead of the functions in torch_scatter.

#### Step-by-Step Installation
Alternatively, you can install all required packages as follows:
```shell
# download ecsg
git clone https://github.com/haozou-csu/ECSG
cd ECSG

# create environment named coldstartcpi
conda create -n ecsg python=3.8.0

# then the environment can be activated to use
conda activate ecsg

# Install pytorch according to hardware
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

# ECSG requires torch-scatter, pip install it with
pip install https://data.pyg.org/whl/torch-1.13.0%2Bcu116/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl

# Install pymatgen and matminer
pip install pymatgen matminer

# Install other tools in requirements.txt
pip install -r requirements.txt

```

#### System Requirements
Recommended Hardware: 128 GB RAM, 40 CPU processors, 4 TB disk storage, 24 GB GPU 

Recommended OS: Linux (Ubuntu 16.04, CentOS 7, etc.)

## Demo data

You can train a demo model by:

```shell
python train.py --name demo --path data/datasets/demo_mp_data.csv --epochs 10
```
If the following files exist in the **models** directory, it means the program runs successfully: 

```text
models
├── demo_meta_model.pkl
├── Roost_demo_0
├── Roost_demo_1
├── Roost_demo_2
├── Roost_demo_3
├── Roost_demo_4
├── ECCNN_demo_0.pth
├── ECCNN_demo_1.pth
├── ECCNN_demo_2.pth
├── ECCNN_demo_3.pth
├── ECCNN_demo_4.pth
├── Magpie_demo_0.json
├── Magpie_demo_1.json
├── Magpie_demo_2.json
├── Magpie_demo_3.json
├── Magpie_demo_4.json
```

## Input
This project provides two feature processing schemes,  each with different input requirements.
#### 1、Feature Processing at Runtime
In this scheme, users need to provide a CSV file containing the materials-id and composition. The program will process the CSV file and generate features at runtime.

The input CSV file must contain the following columns:

- material-id: Unique identifier for each material.
- composition: Chemical composition of the material.

Example of a valid CSV file:

| material-id | composition | 
|-------------|-------------|
| 1           | Fe2O3       | 
| 2           | Al2O3       | 
| ...         | ...         | 


#### 2、Load Preprocessed Feature File
Given the large datasets, feature construction can be quite time-consuming. Additionally, the training process involves cross-validation, which can further increase the computation time.
Therefore, we provide a solution that can load features locally to save time.

You can extract features once and save them using `feature.py`. Run the following command to save the features:
```shell
python feature.py --path your_data.csv --feature_path feature_file
```
For more details, please refer to [Usage](#usage).

## Output
The prediction results will be saved in the **results/meta** folder under the filename **f'{name}_predict_results.csv'**, where **{name}** corresponds to the name of your customized model name. The stability prediction results will be in the target column of the CSV file.


## Usage
#### Prediction
To predict the thermodynamic stability of materials, You can download the pre-trained model files from the following link:

[Download Pre-trained Model](https://drive.google.com/drive/folders/12KcFrYxGNUhQlRy_br0vs98mMsSg-eF0?usp=sharing)

Place all the downloaded model files in the **models** folder in the project root directory.
Use the following command to make predictions. Replace **your_data.csv** with the path to your data file containing the compounds of interest:
```shell
python predict.py --name jarvis_3d --path your_data.csv
```
or use local feature file:
```shell
python predict.py --name jarvis_3d --path your_data.csv --load_from_local 1 --feature_path feature_file
```


## Experiment reproduction

#### Reproducibility with training
Run the following command to start training:

```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv
```
In the **data/datasets** folder, there are [instructions](https://github.com/HaoZou-csu/ECSG/blob/main/data/datasets/readme.md) to download all the datasets in this study. Ensure that the model takes **input** in the form CSV files with materials-ids, composition strings and target values as the columns.

| material-id | composition | target |
|-------------|-------------|--------|
| 1           | Au1Cu1Tm2   | False  |
| 2           | Eu5F1O12P3  | True   |
| ...         | ...         | ...    |

After training, the training log will be saved in the **log** folder through tensorboard, the files containing models' structures and learned parameters will be saved in the **models** folder and the save folder, and the test results will be printed out and saved in **results** folder.

If set `performance_test=True` and a test dataset is defined, the performance of the model will be printed as follows:
```text
        Performance Metrics:
        ====================
        Accuracy: 0.9697802197802198
        Precision: 0.8252212389380531
        Recall: 0.8126361655773421
        F1 Score: 0.8188803512623489
        False Negative Rate (FNR): 0.017172523961661343
        AUC Score: 0.9918534811556208
        AUPR: 0.9139832154286772
        Max F1: 0.8400412796697626
```
#### Train under different data size

You can set the --train_data_used parameter to specify the proportion of the training set to use.
```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv --train_data_used 0.6
```
Please type `python train.py --h` for more help.
```shell
usage: train.py [-h] [--path PATH] [--epochs EPOCHS] [--batchsize BATCHSIZE] [--train TRAIN] [--name NAME] [--train_data_used TRAIN_DATA_USED] [--device DEVICE] [--folds FOLDS] [--lr LR] [--save_model SAVE_MODEL] [--load_from_local LOAD_FROM_LOCAL] [--feature_path FEATURE_PATH]
                [--prediction_model PREDICTION_MODEL] [--train_meta_model TRAIN_META_MODEL] [--performance_test PERFORMANCE_TEST]

Training script for the machine learning model

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the dataset (default: data/datasets/demo_mp_data.csv
  --epochs EPOCHS       Number of epochs to train the model, default: 100
  --batchsize BATCHSIZE
                        Batch size for training (default: 2048)
  --train TRAIN         whether to train the model, 1: true, 0: false, default: 1
  --name NAME           Name of the experiment or model
  --train_data_used TRAIN_DATA_USED
                        Fraction of training data to be used
  --device DEVICE       Device to run the training on, e.g., 'cuda:0' or 'cpu', default: 'cuda:0'
  --folds FOLDS         Number of folds for training ECSG, default: 5
  --lr LR               Learning rate for the optimizer (default: 0.001)
  --save_model SAVE_MODEL
                        Whether to save trained models , 1: true, 0: false, default: 1
  --load_from_local LOAD_FROM_LOCAL
                        Load features from local or generate features from scratch , 1: true, 0: false, default: 0
  --feature_path FEATURE_PATH
                        Path to processed features, default: None
  --prediction_model PREDICTION_MODEL
                        Train a model for predicting or testing , 1: true, 0: false, default: 0
  --train_meta_model TRAIN_META_MODEL
                        Train a single model or train the ensemble model , 1: true, 0: false, default: 1
  --performance_test PERFORMANCE_TEST
                        Whether to test the performance of trained model , 1: true, 0: false, default: 1
```
## Prediction with structure information
To further improve prediction accuracy, we provide modules that include structural information in ECSGs when the structure is known. Follow the steps:
#### 1. Specify the CIF Files Folder
You need to provide a folder containing the CIF files for the materials you want to predict. In this folder, there must also be an id_prop.csv file. This file should include a column that lists the IDs of the CIF files to be used for prediction. We have provided an example in the **data/mp_2024_test/cif** folder.
#### 2. Download and Place Pre-trained Models
Download the pre-trained models and place them in the models folder. Copy the **MP_cif_train_1** folder and the folders from **CGCNN** into the root directory of the **models** folder. This ensures that the required model files are available for running predictions.
##### 3. Running the Prediction Script
Use the following command to run the prediction script:
```shell
python predict_with_cifs.py --name <experiment_name> --cif_path <path_to_cif_folder> --cgcnn_model_path <path_to_cgcnn_models>
```
Please type `predict_with_cifs.py --h` for more help.

```shell
usage: predict_with_cifs.py [-h] [--name NAME] [--cif_path CIF_PATH] [--cgcnn_model_path CGCNN_MODEL_PATH] [--batchsize BATCHSIZE] [--device DEVICE]

Prediction script for structure-based model

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name of the experiment or model (default: MP_cif_train_1)
  --cif_path CIF_PATH   Path to the dataset (default: data/mp_2024_test/cif)
  --cgcnn_model_path CGCNN_MODEL_PATH
                        Path to the dataset (default: models/)
  --batchsize BATCHSIZE
                        Batch size for prediction (default: 2048)
  --device DEVICE       Device to run the training on, e.g., 'cuda:0' or 'cpu', (default: 'cuda:0')


```


## Contact

If any questions, please do not hesitate to contact us at:

Hao Zou, zouhao@csu.edu.cn

Jianxin Wang, jxwang@csu.edu.cn
