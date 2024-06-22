## ECSG

Machine learning offers a promising avenue for expediting the discovery of new compounds by accurately predicting their thermodynamic stability. This approach provides significant advantages in terms of time and resource efficiency compared to traditional experimental and modeling methods. However, most existing models are constructed based on specific domain knowledge, potentially introducing biases that impact their performance. To overcome this limitation, we propose a novel machine learning framework rooted in electron configuration, further enhanced through stack generalization with two additional models grounded in diverse domain knowledge. Experimental results validate the efficacy of our model in accurately predicting the stability of compounds, achieving an impressive Area Under the Curve (AUC) score of 0.988. Notably, our model demonstrates exceptional efficiency in sample utilization, requiring only one-seventh of the data used by existing models to achieve the same performance. 
## Contents
- [Installation](#installation-1)
- [Demo data](#demo-data)
- [Reproducing published results](#reproducing-published-results)
- [Prediction](#prediction)
- [Contact](#contact)

## Installation

#### Installation

- Prerequisites: \
[Python3.*](https://www.python.org/) (version>=3.8)\
[PyTorch](https://pytorch.org/) (version >=1.9.0, <=1.16.0) \
[matminer](https://hackingmaterials.lbl.gov/matminer/)\
[pymatgen](https://pymatgen.org/)


- Dependencies: \
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

Recommended OS: Linux (Ubuntu 16.04, CentOS 7, etc.), Windows 11

## Demo data

You can train a demo model by:

```shell
python train.py --name demo --path data/datasets/demo_mp_data.csv
```

## Reproducing published results

#### Reproducibility with training

```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv
```
The model takes input in the form csv files with materials-ids, composition strings and target values as the columns.

| material-id | composition | target |
|-------------|-------------|--------|
| 1           | Au1Cu1Tm2   | False  |
| 2           | Eu5F1O12P3  | True   |
| ...         | ...         | ...    |

After training, the training log will be saved in the log folder through tensorboard, the files containing models' structures and learned parameters will be saved in the models folder and the save folder, and the test results will be printed out and saved in the results folder.
#### Train under different data size

You can set the --train_data_used parameter to specify the proportion of the training set to use.
```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv --train_data_used 0.6
```

#### Prediction
After the ECSG trained, you can provide a csv file for predicting the stability of custom compounds after the ECSG trained. The format of the file is the same as the input when training the model.

| material-id | composition | target |
|-------------|-------------|--------|
| 1           | Au1Cu1Tm2   | ---    |
| 2           | Eu5F1O12P3  | ---    |
| ...         | ...         | ...    |
The target column can be empty.


For example, after training the demo model, you can enter the following command to predict the thermodynamic stability of the compound you are interested in.
```shell
python ecsg_predict.py --name customized_trained_model_name --path your_data.csv
```



## Contact

If any questions, please do not hesitate to contact us at:

Hao Zou, zouhao@csu.edu.cn

Jianxin Wang, jxwang@csu.edu.cn
