import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import auc,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score, accuracy_score, precision_recall_curve, auc
from xgboost.sklearn import XGBClassifier
import joblib
import pickle
import gc

from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)

from model import  Magpie, Roost,ECCNN, ECCNN_model
from utils.feature_engineering import Magpie_fea, ECCNN_fea

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def load_data(path):
    data = pd.read_csv(path)
    return data

def get_formulas(data):
    formulas = data['composition'].values
    y = data['target'].values
    return formulas, y


def featurization(formulas, index, feature_path= None, load_from_local=False):
    if not load_from_local:

        f_0 = Magpie_fea(formulas)
        f_1 = []
        f_2 = ECCNN_fea(formulas)

        feature = [f_0, f_1, f_2]

    else:
        res = open(feature_path, 'rb')
        data = pickle.load(res)
        f_0 = data[0][index,:]
        f_1 = data[1][index,:]
        f_2 = data[2][index,:]

        feature = [f_0, f_1, f_2]

    return feature


def load_model(path, name, j, i):

    if i == 0:
        m_2_path = 'models/Magpie' + '_' + name + '_' + str(j) + '.json'
        m_2 = XGBClassifier()
        m_2.load_model(m_2_path)
        return m_2


    if i == 1:
        m_5_path = './models/'
        m_5 = Roost(m_5_path, name, j)
        return m_5


    if i == 2:
        m_7_path = 'models/ECCNN'+ '_' + name + '_' + str(j) + '.pth'
        state_dict_7 = torch.load(m_7_path)
        m_7 = ECCNN_model()
        m_7.load_state_dict(state_dict_7)
        return m_7



def bulid_models(path, name, j, save_model):
    m_0 = Magpie(name, j, save_model)
    m_1 = Roost(path, name, j)
    m_2 = ECCNN(name, j, save_model)
    model = [m_0, m_1, m_2]
    return model



def predict_ensemble(save_path, name, model_list, j, data, device='cuda:0', feature_path= None, load_from_local=False ):
    formulas = data['composition'].values
    # y = data['target'].values
    index = data['materials-id'].values
    features = featurization(formulas,index, feature_path, load_from_local)

    pre_y = []
    # for i in range(len(features)):
    for i in model_list:
        m = load_model(save_path, name, j, i)
        # print(i)

        if i == 2:
            batchsize_0 = 8
        else:
            batchsize_0 = 1024

        if i == 0:
            # m = model[i]
            f = features[i]
            y = m.predict_proba(f)[:,1]
            pre_y.append(y)

        elif i == 2:
            with torch.no_grad():
                m.to(device)
                f = features[i]
                f = torch.from_numpy(np.float32(f))

                m.eval()

                predict_data = TensorDataset(f)
                predict_loader = DataLoader(predict_data, batch_size=batchsize_0)

                predict_values = []
                for i, X in enumerate(predict_loader):
                    X = X[0].to(device)
                    y = m(X)
                    predict_values.append(y)
                    torch.cuda.empty_cache()

                y = torch.cat(predict_values, 0)
                y = y.cpu()
                y = y.detach().numpy()
                pre_y.append(y)


        else:
            y = m.predict(data, False, device, j)

            pre_y.append(y)
            torch.cuda.empty_cache()

    pre_y = [n.reshape(-1,1) for n in pre_y]
    return pre_y





def ecsg_predict(name, data, model_list=[0,1,2], folds=10, feature_path=None, load_from_local = False):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_ensemble('models', name, model_list, i, data, feature_path, load_from_local)
        pre_test_y.append(pre_test_y_i)

    pre_test_y = np.array(pre_test_y)
    pre_test = np.mean(pre_test_y, axis=0)

    pre_test = np.swapaxes(pre_test, axis1=0, axis2=1)
    pre_test = pre_test.squeeze(axis=2)

    model = joblib.load(f'models/{name}_meta_model.pkl')
    results = model.predict(pre_test)

    return results

def ecsg_composition_predict(name, data, model_list=[0,1,2], folds=10, feature_path=None, load_from_local = False, device='cuda:0'):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_ensemble('models', name, model_list, i, data,device, feature_path, load_from_local)

        pre_test_y.append(pre_test_y_i)

    pre_test_y = np.array(pre_test_y)
    pre_test = np.mean(pre_test_y, axis=0)

    pre_test = np.swapaxes(pre_test, axis1=0, axis2=1)
    pre_test = pre_test.squeeze(axis=2)


    return pre_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv')
    parser.add_argument("--name", type=str, help="Name of the experiment or model")
    parser.add_argument("--load_from_local", type=int, default=0,
                        help="Load features from local or generate features from scratch , 1: true, 0: false, default: 0")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Path to processed features, default: None")


    args = parser.parse_args()
    name = args.name

    '''data_path'''
    path = args.path
    predict_data = pd.read_csv(path)

    results = ecsg_predict(name, predict_data, folds=5)
    predict_data = predict_data.rename(columns={'target': 'pre_y'})
    predict_data['pre_y'] = results
    
    results = [True if results[n] > 0.5 else False for n in range(len(results))]
    predict_data['pre_y_01'] = results
    
    save_path = 'results/' + name + '_predict_results.csv'
    predict_data.to_csv(save_path, index=False)
    print(f'Prediction results saved in {save_path}')


