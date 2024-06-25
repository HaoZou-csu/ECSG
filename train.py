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


def featurization(formulas, index, feature_path = None, load_from_local=False):
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
        m_0_path = 'models/Magpie' + '_' + name + '_' + str(j) + '.json'
        m_0 = XGBClassifier()
        m_0.load_model(m_0_path)
        return m_0


    if i == 1:
        m_1_path = './models/'
        m_1 = Roost(m_1_path, name, j)
        return m_1


    if i == 2:
        m_2_path = 'models/ECCNN'+ '_' + name + '_' + str(j) + '.pth'
        state_dict_2 = torch.load(m_2_path)
        m_2 = ECCNN_model()
        m_2.load_state_dict(state_dict_2)
        return m_2



def bulid_models(path, name, j, save_model):
    m_0 = Magpie(name, j, save_model)
    m_1 = Roost(path, name, j)
    m_2 = ECCNN(name, j, save_model)
    model = [m_0, m_1, m_2]
    return model

def train_ensemble(data, weight, name, model_list, n_fold, device, lr, criterion, writer, batchsize, epoch, folds=10, random_seed_3=123, save_model=True, feature_path=None, load_from_local=False):
    formulas = data['composition'].values
    y = data['target'].values

    index = data['materials-id'].values

    train_for = index
    train_y = y

    kfolds = KFold(n_splits=folds, shuffle=True, random_state=random_seed_3)

    j = 0
    for train, val in kfolds.split(train_for):
        if j == n_fold:
            train_cv_for = train_for[train]
            val_cv_for = train_for[val]
            # train_cv_y = train_y[train]
            # val_cv_y = train_y[val]

            train_cv_X = featurization(formulas[train], train_cv_for, feature_path, load_from_local)
            val_cv_X = featurization(formulas[val], val_cv_for, feature_path, load_from_local)

            train_cv_weight = weight[train]

            models = bulid_models(path, name, j, save_model=save_model)
            for i in model_list:
                train_cv_X_i = train_cv_X[i]
                val_cv_X_i = val_cv_X[i]
                train_cv_y = train_y[train]
                val_cv_y = train_y[val]
                model_i = models[i]

                if i == 2:
                    batchsize_0 = 32
                else:
                    batchsize_0 = batchsize

                if i == 1:
                    print('/n'
                        '======================Train Roost========================\n')
                    model_i.train(data, False,  device, epoch, batchsize, n_fold, folds,  random_seed_3)

                elif i == 0:
                    print('/n'
                        '======================Train Magpie========================\n')
                    model_i.train(train_cv_X_i, train_cv_y, train_cv_weight)
                else:
                    print('/n'
                        '======================Train ECCNN========================\n')
                    train_cv_X_i = torch.from_numpy(np.float32(train_cv_X_i))
                    train_cv_y = train_cv_y.astype(np.float32)
                    train_cv_y = torch.from_numpy(train_cv_y.reshape(-1, 1))

                    val_cv_X_i = torch.from_numpy(np.float32(val_cv_X_i))
                    val_cv_y = val_cv_y.astype(np.float32)
                    val_cv_y = torch.from_numpy(val_cv_y.reshape(-1, 1))

                    train_cv_weight_i = torch.from_numpy(train_cv_weight.reshape(-1, 1))

                    train_dataset = TensorDataset(train_cv_X_i, train_cv_y, train_cv_weight_i)
                    train_loader = DataLoader(train_dataset, batch_size=batchsize_0)
                    val_dataset = TensorDataset(val_cv_X_i, val_cv_y)
                    val_loader = DataLoader(val_dataset, batch_size=batchsize_0)
                    model_i.trainer(device, train_loader, val_loader, lr=lr,criterion=criterion, writer=writer, epochs=epoch)
                torch.cuda.empty_cache()

        j = j + 1

def train_meta(X, y, name, save_model=True):
    model = LinearRegression(positive=True)
    model = LinearRegression()
    model.fit(X,y)
    joblib.dump(model, f'models/{name}_meta_model.pkl')

def predict_ensemble(save_path, name, model_list, j, data, device='cuda:0', feature_path=None, load_from_local=False):
    formulas = data['composition'].values
    y = data['target'].values
    index = data['materials-id'].values
    features = featurization(formulas, index, feature_path, load_from_local)

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
            y = m.predict(data, False, device,  j)

            pre_y.append(y)
            torch.cuda.empty_cache()

    pre_y = [n.reshape(-1,1) for n in pre_y]
    return pre_y


def each_fold_path(data, n_fold, folds=10, random_seed_3=123):
    train_for = data
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_seed_3)
    i = 0
    for train, test in kf.split(train_for):
        if i == n_fold:
            test = train_for.index[test]
            test = np.array(test).tolist()

            fold_data = train_for.loc[test]
            fold_y = fold_data['target'].values
        i = i + 1
    return fold_data, fold_y



def get_train_data(data, weight, name, model_list, device, lr, criterion, log= True, batchsize=1024, epoch=100, folds=10,  random_seed_3=123, save_model=True, train=True, feature_path= None, load_from_local=False):
    if log:
        writer = SummaryWriter('./log/' + name)
    if train:
        for i in range(folds):
            print(
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                f"--------------Train on fold {i + 1}--------------\n"
                "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            )
            train_ensemble(data, weight, name, model_list, i, device, lr, criterion, writer, batchsize, epoch, folds=folds, random_seed_3=random_seed_3, save_model=save_model, feature_path=feature_path, load_from_local=load_from_local)
    train_data = []
    train_y = []
    for j in range(folds):
        fold_pd, fold_y = each_fold_path(data, j, folds= folds, random_seed_3=random_seed_3)
        pre_y = predict_ensemble('models', name, model_list, j, fold_pd, load_from_local=False)
        pre_y = [n.ravel() for n in pre_y]
        pre_y = np.array(pre_y)
        pre_y = np.swapaxes(pre_y, axis1= 0, axis2=1)
        train_data.append(pre_y)
        train_y.append(fold_y)
    train_data = tuple(train_data)
    train_data = np.vstack(train_data)
    train_y = tuple(train_y)
    train_y = np.hstack(train_y)
    return train_data, train_y

def y_to_01(y):
    new_y = []
    for i in range(len(y)):
        if y[i] > 0.5:
            new_y.append(1)
        else:
            new_y.append(0)
    return np.array(new_y)

def Performance( pre_test_y_prob, test_y):

    test_y = test_y.astype(int)
    pre_test_y = y_to_01(pre_test_y_prob)
    accuracy = accuracy_score(test_y, pre_test_y)

    precision, recall, _ = precision_recall_curve(test_y, pre_test_y_prob)
    aupr = auc(recall, precision)
    max_f1 = max(2 * (precision * recall) / (precision + recall))


    precision = precision_score(test_y, pre_test_y, zero_division=0)
    recall = recall_score(test_y, pre_test_y)
    f1 = f1_score(test_y, pre_test_y)
    fnr = confusion_matrix(test_y, pre_test_y, normalize='pred')[1][0]
    auc_score = roc_auc_score(test_y, pre_test_y_prob)



    return accuracy,precision,recall,f1,fnr,auc_score, aupr,max_f1


def evaluate(name, data, model_list, folds=10):
    pre_test_y = []
    for i in range(folds):
        pre_test_y_i = predict_ensemble('models', name, model_list, i, data)
        pre_test_y.append(pre_test_y_i)

    pre_test_y = np.array(pre_test_y)
    pre_test = np.mean(pre_test_y, axis=0)

    pre_test = np.swapaxes(pre_test, axis1=0, axis2=1)
    pre_test = pre_test.squeeze(axis=2)

    model = joblib.load(f'models/{name}_meta_model.pkl')
    results = model.predict(pre_test)

    target_y = data['target'].values

    performance = Performance(results, target_y)
    return pre_test, performance, results


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Training script for the machine learning model")

    # Add arguments to the parser
    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv',
                        help="Path to the dataset (default: data/datasets/demo_mp_data.csv")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train the model, default: 100")
    parser.add_argument("--batchsize", type=int, default=2048,
                        help="Batch size for training (default: 2048)")
    parser.add_argument("--train", type=int, default=1,
                        help="whether to train the model, 1: true, 0: false, default: 1")
    parser.add_argument("--name", type=str, help="Name of the experiment or model")
    parser.add_argument("--train_data_used", type=float, default=1.0,
                        help="Fraction of training data to be used")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to run the training on, e.g., 'cuda:0' or 'cpu', default: 'cuda:0'")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds for training ECSG, default: 5")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for the optimizer (default: 0.001)")
    parser.add_argument("--save_model", type=int, default=1,
                        help="Whether to save trained models , 1: true, 0: false, default: 1")
    parser.add_argument("--load_from_local", type=int, default=0,
                        help="Load features from local or generate features from scratch , 1: true, 0: false, default: 0")
    parser.add_argument("--feature_path", type=str, default=None,
                        help="Path to processed features, default: None")
    parser.add_argument("--prediction_model", type=int, default=0,
                        help="Train a model for predicting or testing , 1: true, 0: false, default: 0")
    parser.add_argument("--train_meta_model", type=bool, default=True,
                        help="Train a single model or train the ensemble model , 1: true, 0: false, default: 1")
    parser.add_argument("--performance_test", type=bool, default=True,
                        help="Whether to test the performance of trained model , 1: true, 0: false, default: 1")

    args = parser.parse_args()
    device = args.device
    print(device)

    """tasks type"""
    train = args.train
    # print(train)
    save_model = args.save_model   ## wheather to save trained models
    load_from_local = args.load_from_local  ## load features from local  or generate features from scrath
    prediction_model = args.prediction_model  ## train a model for predicting or testing
    train_meta_model = args.train_meta_model   ## train a single model or train the ensemble model
    performance_test = args.performance_test

    model_list = [0,1,2]


    """hyperparameters"""
    criterion = torch.nn.BCELoss(reduction='sum')
    batchsize = args.batchsize
    lr = args.lr
    epoch = args.epochs
    name = args.name
    folds = args.folds
    train_data_used = args.train_data_used


    write = SummaryWriter('log/'+ name)
    '''data_path'''
    path = args.path
    data = pd.read_csv(path)

    """select seed"""
    random_seed_1 = 123
    random_seed_2 = 2
    random_seed_3 = 123


    train_X, test_X, _, _ = train_test_split(data, data, test_size=0.1, random_state=random_seed_1)
    if train_data_used < 1:
        train_X, U_X, _, _ = train_test_split(train_X, train_X, train_size=train_data_used, random_state=random_seed_2)
    # print(train_X.shape)
    if prediction_model:
        train_X = data
        test_X = pd.read_csv('data/datasets/test_X.csv')

    if train:
        weight = np.ones(len(train_X)) / len(train_X)
        train_data, train_y = get_train_data(train_X, weight, name, model_list, device, lr, criterion, True, batchsize=1024, epoch=epoch, folds=folds,  random_seed_3=random_seed_3, save_model=save_model, train=True)

        if train_meta_model:
            train_meta(train_data, train_y, name)
            # if not os.path.exists('save/data/'+ name):
            #     os.mkdir('save/data/'+ name)
            # np.save('save/data/'+ name +'/train_data.npy', train_data)
            # np.save('save/data/'+ name +'/train_y.npy', train_y)

    if performance_test:
        pre_test, performance, results = evaluate(name, test_X, model_list, folds=folds)
        accuracy, precision, recall, f1, fnr, auc_score, aupr, max_f1 = performance


        print(f"""
        Performance Metrics:
        ====================
        Accuracy: {accuracy}
        Precision: {precision}
        Recall: {recall}
        F1 Score: {f1}
        False Negative Rate (FNR): {fnr}
        AUC Score: {auc_score}
        AUPR: {aupr}
        Max F1: {max_f1}
        """)
