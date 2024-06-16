## ECSG

Machine learning offers a promising avenue for expediting the discovery of new compounds by accurately predicting their thermodynamic stability. This approach provides significant advantages in terms of time and resource efficiency compared to traditional experimental and modeling methods. However, most existing models are constructed based on specific domain knowledge, potentially introducing biases that impact their performance. To overcome this limitation, we propose a novel machine learning framework rooted in electron configuration, further enhanced through stack generalization with two additional models grounded in diverse domain knowledge. Experimental results validate the efficacy of our model in accurately predicting the stability of compounds, achieving an impressive Area Under the Curve (AUC) score of 0.988. Notably, our model demonstrates exceptional efficiency in sample utilization, requiring only one-seventh of the data used by existing models to achieve the same performance. 
- [Installation](#Installation)
- [Demo data](#Demo-data)
- [Reproducing published results](#reproducing-published-results)
- [Prediction](#Prediction)
- [Contact](#Contact)

## Installation

#### Installation

- Prerequisites: \
[Python3.*](https://www.python.org/) (version>=3.8)\
[matminer](https://hackingmaterials.lbl.gov/matminer/)\
[pymatgen](https://pymatgen.org/)


- Dependencies: \
numpy==1.23.5\
pandas==1.5.2\
scikit-learn==1.2.0\
torch==1.13.0+cu116\
tensorboard==2.14.0\
torch_geometric==2.4.0\
torch-scatter==2.0.9\
tqdm==4.64.1\
xgboost==2.0.3\
\
scipy\
pytest


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

# install other tools in requirements.txt
pip install -r requirements.txt

```

#### System Requirements
Recommended Hardware: 128 GB RAM, 40 CPU processors, 4 TB disk storage, >=30 GB GPU 

Recommended OS: Linux (Ubuntu 16.04, CentOS 7, etc.), Windows 11

## Demo data

You can train a demo model by:

```shell
python train.py --name customized_model_name --path data/datasets/demo_mp_data.csv
```

## Reproducing published results

### Reproducibility with training

```shell
python train.py --name customized_model_name --path data/datasets/mp_data.csv
```

### Reproducibility without training

### Prediction
python train.py --name customized_trained_model_name --path your_data_path

### Fluidity of Materials Project
Please note that the Materials Project database is constantly changing. While this doesn't present any issues for the direct replication of our results or the application of new models trained on formation energy, it may complicate the strict replication of our results for models trained on multiple properties (e.g., band gap and formation energy learned simultaneously).

## Contact

If any questions, please do not hesitate to contact us at:

Hao Zou, zouhao@csu.edu.cn

Jianxin Wang, jxwang@csu.edu.cn
