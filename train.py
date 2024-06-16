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

from model import Elfrac, Magpie, Meredig, ElemNet, Roost, ATCNN, ECCNN, ECCNN_model, ATCNN_model, ElemNet_model
from utils.feature_engineering import Elfrac_fea, Magpie_fea, Meredig_fea, ElemNet_fea, ATCNN_fea, ECCNN_fea

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def load_data(path):
    data = pd.read_csv(path)
    return data

def get_formulas(data):
    formulas = data['composition'].values
    y = data['target'].values
    return formulas, y


def featurization(formulas, index, load_from_local=False):
    if not load_from_local:
        # f_1 = Elfrac_fea(formulas)
        # f_2 = Magpie_fea(formulas)
        # f_3 = Meredig_fea(formulas)
        # f_4 = ElemNet_fea(formulas)
        # f_5 = []
        # f_6 = ATCNN_fea(formulas)
        # f_7 = ECCNN_fea(formulas)

        f_1 = []
        f_2 = []
        f_3 = []
        f_4 =  ElemNet_fea(formulas)
        f_5 = []
        f_6 = []
        f_7 = []
        feature = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]

    else:
        res = open("D:/programming/ECSG_git/data/file", 'rb')
        data = pickle.load(res)
        f_1 = data[0][index,:]
        f_2 = data[1][index,:]
        f_3 = data[2][index,:]
        f_4 = data[3][index,:]
        f_5 = []
        f_6 = data[5][index,:]
        f_7 = data[6][index,:]

        feature = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]

    return feature


def load_model(path, name, j, i):
    if i == 0:
        m_1_path = 'save/Elfrac' + '_' + name + '_' + str(j) + '.json'
        m_1 = XGBClassifier()
        m_1.load_model(m_1_path)
        return m_1

    if i == 1:
        m_2_path = 'save/Magpie' + '_' + name + '_' + str(j) + '.json'
        m_2 = XGBClassifier()
        m_2.load_model(m_2_path)
        return m_2
        # m_2_path = 'save/Magpie' + '_' + name + '_' + str(j) + '.joblib'
        # m_2 = joblib.load(m_2_path)
        # return m_2

    if i == 2:
        m_3_path = 'save/Meredig' + '_' + name + '_' + str(j) + '.json'
        m_3 = XGBClassifier()
        m_3.load_model(m_3_path)
        return m_3

    if i == 3:
        m_4_path = 'save/ElemNet' + '_' + name + '_' + str(j) + '.best'
        state_dict_4 = torch.load(m_4_path)
        m_4 = ElemNet_model()
        m_4.load_state_dict(state_dict_4)
        return m_4

    if i == 4:
        m_5_path = './models/'
        m_5 = Roost(m_5_path, name, j)
        return m_5

    if i == 5:
        m_6_path = 'save/ATCNN' + '_' + name + '_' + str(j) + '.best'
        state_dict_6 = torch.load(m_6_path)
        m_6 = ATCNN_model()
        m_6.load_state_dict(state_dict_6)
        return m_6

    if i == 6:
        m_7_path = 'save/ECCNN'+ '_' + name + '_' + str(j) + '.best'
        state_dict_7 = torch.load(m_7_path)
        m_7 = ECCNN_model()
        m_7.load_state_dict(state_dict_7)
        return m_7



def bulid_models(path, name, j, save_model):
    m_1 = Elfrac(name, j, save_model)
    m_2 = Magpie(name, j, save_model)
    m_3 = Meredig(name, j, save_model)
    m_4 = ElemNet(name, j, save_model)
    m_5 = Roost(path, name, j)
    m_6 = ATCNN(name, j, save_model)
    m_7 = ECCNN(name, j, save_model)
    model = [m_1, m_2, m_3, m_4, m_5, m_6, m_7]
    return model

def train_ensemble(data, weight, name, model_list, n_fold, device, lr, criterion, writer, batchsize, epoch, folds=10, random_seed_3=123, save_model=True):
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

            train_cv_X = featurization(formulas[train], train_cv_for, True)
            val_cv_X = featurization(formulas[val], val_cv_for, True)

            train_cv_weight = weight[train]

            models = bulid_models(path, name, j, save_model=save_model)
            # for i in range(len(train_cv_X)):
            for i in model_list:
                train_cv_X_i = train_cv_X[i]
                val_cv_X_i = val_cv_X[i]
                train_cv_y = train_y[train]
                val_cv_y = train_y[val]
                model_i = models[i]

                if i == 6:
                    batchsize_0 = 32
                else:
                    batchsize_0 = batchsize

                if i == 4:
                    model_i.train(data, False,  n_fold, folds,  random_seed_3) ### 原来的函数需要改参数
                # elif i == 6:
                #     train_dataset = TensorDataset(train_cv_X_i, train_cv_y)
                #     train_loader = DataLoader(train_dataset, batch_size=32)
                #     val_dataset = TensorDataset(val_cv_X_i, val_cv_y)
                #     val_loader = DataLoader(val_dataset, batch_size=32)
                #     model_i.trainer(device, train_loader, val_loader, lr=lr,criterion=criterion, epochs=epoch)
                elif i < 3:
                    model_i.train(train_cv_X_i, train_cv_y, train_cv_weight)
                else:
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
    joblib.dump(model, f'save/{name}_meta_model.pkl')

def predict_ensemble(save_path, name, model_list, j, data, device='cuda:0'):
    formulas = data['composition'].values
    y = data['target'].values
    index = data['materials-id'].values
    features = featurization(formulas,index, True)

    pre_y = []
    # for i in range(len(features)):
    for i in model_list:
        m = load_model(save_path, name, j, i)
        # print(i)

        if i == 6:
            batchsize_0 = 8
        else:
            batchsize_0 = 1024

        if i < 3:
            # m = model[i]
            f = features[i]
            y = m.predict_proba(f)[:,1]
            pre_y.append(y)

        elif i != 4:
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

        # elif i != 4:
        #     # m = model[i]
        #     f = features[i]
        #     f = torch.from_numpy(np.float32(f))
        #     # pre_gene = PredictDataGenerator(f, 32)
        #     y = m(f)
        #
        #     y = y.detach().numpy()
        #     pre_y.append(y)
        #     torch.cuda.empty_cache()


        else:
            y = m.predict(data, False, j)

            pre_y.append(y)
            torch.cuda.empty_cache()

    pre_y = [n.reshape(-1,1) for n in pre_y]
    return pre_y



def proportional_selection(U_X, pre_U_y_prob, n):
    '''

    Args:
        U_X:
        pre_U_y_prob:
        n: n is the portion of selected sample

    Returns:

    '''
    assert U_X.shape[0] == pre_U_y_prob.shape[0]

    selected_number = int(U_X.shape[0] * n)
    class_1_pre_U_y_prob = pre_U_y_prob.ravel()
    rank_index = np.argsort(class_1_pre_U_y_prob)
    ##从小到大预测为1的概率
    class_0_index = rank_index[:selected_number]
    class_1_index = rank_index[-selected_number:]
    rest_index = rank_index[selected_number: -selected_number]

    selected_U_X_0 = U_X[class_0_index,:]
    selected_U_X_1 = U_X[class_1_index,:]
    rest_U_X = U_X[rest_index]
    return selected_U_X_0, selected_U_X_1, rest_U_X

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



def get_train_data(data, weight, name, model_list, device, lr, criterion, log= True, batchsize=1024, epoch=100, folds=10,  random_seed_3=123, save_model=True, train=True):
    if log:
        writer = SummaryWriter('./log/' + name)
    if train:
        for i in range(folds):
            train_ensemble(data, weight, name, model_list, i, device, lr, criterion, writer, batchsize, epoch, folds=folds, random_seed_3=random_seed_3, save_model=save_model)
    train_data = []
    train_y = []
    for j in range(folds):
        fold_pd, fold_y = each_fold_path(data, j, folds= folds, random_seed_3=random_seed_3)
        pre_y = predict_ensemble('save', name, model_list, j, fold_pd)
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
        pre_test_y_i = predict_ensemble('save', name, model_list, i, data)
        pre_test_y.append(pre_test_y_i)

    pre_test_y = np.array(pre_test_y)
    pre_test = np.mean(pre_test_y, axis=0)

    pre_test = np.swapaxes(pre_test, axis1=0, axis2=1)
    pre_test = pre_test.squeeze(axis=2)

    model = joblib.load(f'save/{name}_meta_model.pkl')
    results = model.predict(pre_test)

    target_y = data['target'].values

    performance = Performance(results, target_y)
    return pre_test, performance, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--train", type=bool, default=True, help='train or not')
    parser.add_argument("--name", type=str)

    args = parser.parse_args()
    print(args.name)

    """tasks type"""
    train = True
    save_model = True
    load_from_local = False
    prediction_model = False
    train_single = False

    """hyperparameters"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCELoss(reduction='sum')
    batchsize = args.batchsize
    lr = 1e-3
    epoch = args.epochs
    name = args.name
    model_list = [1,4,6]
    folds = 5


    # write = SummaryWriter('log/'+ name)
    '''data_path'''
    path = args.path
    data = pd.read_csv(path)

    """select seed"""
    random_seed_1 = 123
    random_seed_2 = 2
    random_seed_3 = 123
    train_data_used = 0.2

    train_X, test_X, _, _ = train_test_split(data, data, test_size=0.1, random_state=random_seed_1)
    train_X, U_X, _, _ = train_test_split(train_X, train_X, train_size=train_data_used, random_state=random_seed_2)

    # print(train_X.shape)
    if prediction_model:
        train_X = data
        test_X = pd.read_csv('data/datasets/test_tm_3.csv')

    if train:
        weight = np.ones(len(train_X)) / len(train_X)
        train_data, train_y = get_train_data(train_X, weight, name, model_list, device, lr, criterion, True, batchsize=1024, epoch=epoch, folds=folds,  random_seed_3=random_seed_3, save_model=save_model, train=True)

        if not train_single:
            train_meta(train_data, train_y, name)
            if not os.path.exists('save/data/'+ name):
                os.mkdir('save/data/'+ name)
            np.save('save/data/'+ name +'/train_data.npy', train_data)
            np.save('save/data/'+ name +'/train_y.npy', train_y)

    # test_X = pd.read_csv('data/datasets/jarvis_2021_Roost.csv')
    # pre_train, performance_T, results_T = evaluate(name, train_X)

    pre_test, performance, results = evaluate(name, test_X, model_list, folds=folds)
    print(performance) #0.9706959706959707, 0.8201284796573876, 0.8344226579520697, 0.8272138228941684, 0.015221309833767275, 0.9935485669750582
    # pre_test_U, performance_U, results_U = evaluate(name, U_X)

    # np.save('data/'+ name +'/train_data.npy', pre_train)
    # np.save('save/data/'+ name +'/test_data.npy', pre_test)
    # np.save('save/data/'+ name +'/U_data.npy', pre_test_U)

    # (0.8581387808041504, 0.42017879948914433, 0.43866666666666665, 0.4292237442922375, 0.07818012999071496,
    #  0.8169767441860465)


    ## 315490304
    from sklearn.metrics import confusion_matrix