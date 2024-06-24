## ECSG

Machine learning offers a promising avenue for expediting the discovery of new compounds by accurately predicting their thermodynamic stability. This approach provides significant advantages in terms of time and resource efficiency compared to traditional experimental and modeling methods. However, most existing models are constructed based on specific domain knowledge, potentially introducing biases that impact their performance. To overcome this limitation, we propose a novel machine learning framework rooted in electron configuration, further enhanced through stack generalization with two additional models grounded in diverse domain knowledge. Experimental results validate the efficacy of our model in accurately predicting the stability of compounds, achieving an impressive Area Under the Curve (AUC) score of 0.988. Notably, our model demonstrates exceptional efficiency in sample utilization, requiring only one-seventh of the data used by existing models to achieve the same performance. 
## Contents
- [Installation](#installation-1)
- [Prediction](#prediction)
- [Demo data](#demo-data)
- [Experiment reproduction](#Experiment reproduction)
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

## Prediction
To predict the thermodynamic stability of materials, You can download the pre-trained model files from the following link:

[Download Pre-trained Model](https://drive.google.com/drive/folders/12KcFrYxGNUhQlRy_br0vs98mMsSg-eF0?usp=sharing)

Place all the downloaded model files in the **models** folder in the project root directory.
Use the following command to make predictions. Replace **your_data.csv** with the path to your data file containing the compounds of interest:
```shell
python predict.py --name jarvis_3d --path your_data.csv
```
The input CSV file must contain the following columns:

- material-id: Unique identifier for each material.
- composition: Chemical composition of the material.

Example of a valid CSV file:

| material-id | composition | 
|-------------|-------------|
| 1           | Fe2O3       | 
| 2           | Al2O3       | 
| ...         | ...         | 

The prediction results will be saved in the **results/meta** folder under the filename **f'{name}_predict_results.csv'**, where **{name}** corresponds to the name of your customized model name. The stability prediction results will be in the target column of the CSV file.

## Experiment reproduction

#### Reproducibility with training
Run the following command to start training:

```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv
```
In the **data/datasets** folder, there are instructions to download all the datasets in this study. Ensure that the model takes **input** in the form CSV files with materials-ids, composition strings and target values as the columns.

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
        Accuracy: 0.67
        Precision: 0.6923076923076923
        Recall: 0.5625
        F1 Score: 0.6206896551724138
        False Negative Rate (FNR): 0.3442622950819672
        AUC Score: 0.8004807692307693
        AUPR: 0.7195406377125553
        Max F1: 0.7868852459016393
```
#### Train under different data size

You can set the --train_data_used parameter to specify the proportion of the training set to use.
```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv --train_data_used 0.6
```
Please type `python train.py --h` for more help.
```shell
usage: train.py [-h] [--path PATH] [--epochs EPOCHS] [--batchsize BATCHSIZE] [--train TRAIN] [--name NAME] [--train_data_used TRAIN_DATA_USED] [--device DEVICE] [--folds FOLDS] [--lr LR] [--save_model SAVE_MODEL] [--load_from_local LOAD_FROM_LOCAL] [--prediction_model PREDICTION_MODEL]
                [--train_meta_model TRAIN_META_MODEL]

Training script for the machine learning model

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the dataset (default: data/datasets/demo_mp_data.csv
  --epochs EPOCHS       Number of epochs to train the model (default: 100)
  --batchsize BATCHSIZE
                        Batch size for training (default: 2048)
  --train TRAIN         Boolean flag to indicate whether to train the model, default: True
  --name NAME           Name of the experiment or model
  --train_data_used TRAIN_DATA_USED
                        Fraction of training data to be used
  --device DEVICE       Device to run the training on, e.g., 'cuda:0' or 'cpu'
  --folds FOLDS         Number of folds for training ECSG (default: 5)
  --lr LR               Learning rate for the optimizer (default: 0.001)
  --save_model SAVE_MODEL
                        Whether to save trained models (default: True)
  --load_from_local LOAD_FROM_LOCAL
                        Load features from local or generate features from scratch (default: False)
  --prediction_model PREDICTION_MODEL
                        Train a model for predicting or testing (default: False)
  --train_meta_model TRAIN_META_MODEL
                        Train a single model or train the ensemble model (default: True)

```
#### Improve the efficiency of feature construction
Given the large datasets, feature construction can be quite time-consuming. Additionally, the training process involves cross-validation, which can further increase the computation time.

You can extract features once and save them using `feature.py`. Run the following command to save the features:
```shell
python feature.py --path your_data.csv
```

Then load the saved features during training by setting the `--load_from_local` flag to True in `train.py`. This will load the features from the local storage instead of extracting them again, saving significant time:


## Contact

If any questions, please do not hesitate to contact us at:

Hao Zou, zouhao@csu.edu.cn

Jianxin Wang, jxwang@csu.edu.cn
