import pandas as pd
import numpy as np
from time import time


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from sklearn.model_selection import train_test_split, KFold
from xgboost.sklearn import XGBClassifier

import sys
from os.path import dirname, abspath
path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)



from sklearn.metrics import auc,recall_score,f1_score,precision_score,confusion_matrix,roc_auc_score, accuracy_score
from roost.roost_train import model_5_predict, main, train_model_5
import gc






class Elfrac():
    def __init__(self, name, number, save_model=True):
        self.model = XGBClassifier()
        self.model_name = 'Elfrac' + '_' + name + '_' +  str(number)
        self.save_path = './models/' + self.model_name + '.json'
        self.save_model = save_model

    def train(self, X, y, weight):
        '''

        Args:
            X: Features
            y: Lable

        '''
        self.model.fit(X, y)
        if self.save_model:
            self.model.save_model(self.save_path)
            print('Model file saved to '+ self.save_path)

    def performance(self, test_X, test_y):
        pre_test_y = self.model.predict(test_X)
        pre_test_y_prob = self.model.predict_proba(test_X)[:, 1]

        accuracy = accuracy_score(test_y, pre_test_y)
        precision = precision_score(test_y, pre_test_y, zero_division=0)
        recall = recall_score(test_y, pre_test_y)
        f1 = f1_score(test_y, pre_test_y)
        fnr = confusion_matrix(test_y, pre_test_y, normalize='pred')[1][0]
        auc_score = roc_auc_score(test_y, pre_test_y_prob)

        return accuracy,precision,recall,f1,fnr,auc_score

class Magpie(Elfrac):
    def __init__(self, name, number, save_model=True):
        super(Magpie, self).__init__(name, number, save_model)
        self.model_name = 'Magpie' + '_' + name + '_' +  str(number)

        self.model = XGBClassifier()
        self.save_path = './models/' + self.model_name + '.json'  ##json if xgboost   else joblib

        # self.model = RandomForestClassifier()
        # self.save_path = './save/' + self.model_name + '.joblib'   ##json if xgboost   else joblib
    # def train(self, X, y, weight):
    #     '''
    #     Args:
    #         X: Features
    #         y: Lable
    #
    #     '''
    #     from joblib import dump, load
    #     self.model.fit(X, y)
    #     if self.save_model:
    #         # self.model.save_model(self.save_path) ## xgboost
    #         dump(self.model, self.save_path)
    #         print('Model file saved to ' + self.save_path)

class Meredig(Elfrac):
    def __init__(self, name, number, save_model=True):
        super(Meredig, self).__init__(name, number, save_model)
        self.model = XGBClassifier()
        self.model_name = 'Meredig' + '_' + name + '_' +  str(number)
        self.save_path = './models/' + self.model_name + '.json'

class Roost():
    def __init__(self, path, name, number):
        self.model_name = 'Roost' + '_' + name + '_' +  str(number)
        self.save_path = './save/' + self.model_name
        self.data_path = path

    def train(self, df, cuda, device, epoch, batch_size,n_fold, kfolds=10,  random_seed_3=123):
        # path = self.data_path
        model_name = self.model_name
        train_model_5(df, cuda, device, epoch, batch_size, n_fold, model_name, kfolds=kfolds, random_seed_3=random_seed_3)

    def predict(self, df, cuda, n_fold):
        model_name = self.model_name
        df_new = df.reset_index(drop=True)
        model_5_predict(df_new, cuda, n_fold, model_name)
        results = pd.read_csv(f'results/Roost/predict_results_{model_name}_r-0.csv')
        results.rename(columns={'id':'materials-id'}, inplace=True)
        pre_y = pd.merge(df_new, results, on='materials-id')['class-1-pred_0_pro'].values
        return pre_y

class ElemNet_model(nn.Module):
    def __init__(self):
        super(ElemNet_model, self).__init__()

        self.model = nn.Sequential(nn.Linear(86, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 1024),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),

                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1),

                                   nn.Linear(512, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Dropout(0.3),

                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),

                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 1),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        x = self.model(x)
        return x

class ATCNN_model(nn.Module):
    def __init__(self):
        super(ATCNN_model, self).__init__()
        self.input_shape =  (10, 10, 1)

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 200),
            nn.BatchNorm1d(200), nn.Dropout(0.1), nn.ReLU(),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100), nn.Dropout(0.2), nn.ReLU(),
            nn.Linear(100, 1), nn.Sigmoid()
        )

    def forward(self, X):
        X = X.squeeze(-1).unsqueeze(1)
        y = self.model(X)
        return y

class ECCNN_model(nn.Module):
    def __init__(self):
        super(ECCNN_model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(8, 64, 5, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=2),
            nn.Flatten(),
            nn.Linear(308096, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



def get_right_count(output, target):
    count = 0
    for i in range(len(output)):
        if output[i] > 0.5:
            output[i] = 1
        else:
            output[i] = 0

        if output[i] == target[i]:
            count += 1

    return count


class ElemNet():
    def __init__(self, name, number, save_model=True):
        self.model_name = 'ElemNet' + '_' + name + '_' + str(number)
        self.save_path = './models/' + self.model_name
        self.save_model = save_model

    def build_model(self):
        model = ElemNet_model()
        self.model = model

    def train(self, device, train_loader, lr, criterion, epoch):

        size = len(train_loader.dataset)

        self.device = device
        self.model.to(self.device)
        self.model.train()
        train_loss = 0
        right_count = 0

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for batch, (input, target, weight) in enumerate(train_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            weight = weight.to(self.device)

            output = self.model(input)
            s_loss = criterion(output, target)
            loss = (s_loss * weight).mean()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right_count += get_right_count(output, target)
            train_loss += loss * len(input)
            del loss

            # if batch % 100 == 0:
            #     loss, current = loss.item(), batch * len(input[-1])
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = right_count / len(train_loader.dataset)
        # print(f"epoch{epoch}: train_loss  {train_loss:>7f}")

        return train_loss, train_acc


    def valuate(self, test_loader, criterion, min_loss, max_acc):
        # min_loss = 1e-60

        # size = len(test_loader.dataset)
        # num_batches = len(test_loader)
        self.model.eval()
        test_loss, right_count = 0, 0
        with torch.no_grad():
            for batch, (input, target) in enumerate(test_loader):
                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(input)
                test_loss += criterion(output, target).sum()
                right_count += get_right_count(output, target)

            test_loss /= len(test_loader.dataset)
            acc = right_count / len(test_loader.dataset)
            # print(f"test_loss=: {test_loss:>8f} ")
            if test_loss < min_loss:
                min_loss = test_loss
                # if self.save_model:
                #     torch.save(self.model.state_dict(), self.save_path + '.best')

            if acc > max_acc:
                max_acc = acc
            # print(f'acc is {acc}')

        return test_loss, min_loss, acc, max_acc

    def trainer(self, device, train_loader, test_loader, lr, criterion, writer, epochs):
        min_loss = 1e20
        max_acc = -100
        self.build_model()
        for i in range(epochs):
            start_time = time()
            train_loss, train_acc = self.train(device, train_loader, lr, criterion, i)
            end_time = time()
            test_loss, min_loss, acc, max_acc = self.valuate(test_loader, criterion, min_loss, max_acc)


            writer.add_scalar('loss/train_' + self.model_name, train_loss, i)
            writer.add_scalar('acc/train_' + self.model_name, train_acc, i)
            writer.add_scalar('loss/test_' + self.model_name, test_loss, i)
            writer.add_scalar('acc/test_' + self.model_name, acc, i)
            dur = end_time - start_time

            print(f'epoch {i}, time:{dur:>4f}         ==============================================')
            print(f'train_loss is {train_loss:>5f}, test_loss is {test_loss:>5f}, acc is {acc:>3f}, max acc is {max_acc:>3f}')
            if test_loss == min_loss:
                print(f'test_loss < min_loss, save {self.model_name}')
            if self.save_model:
                torch.save(self.model.state_dict(), self.save_path + '.pth')

    def predict(self, X):
        self.model
        X = torch.from_numpy(X)
        X = X.to(self.device)
        y = self.model(X)
        return y

class ATCNN(ElemNet):
    def __init__(self, name, number, save_model):
        super(ATCNN, self).__init__(name, number, save_model)
        self.model_name = 'ATCNN' + '_' + name + '_' + str(number)
        self.save_path = './models/' + self.model_name

    def build_model(self):
        model = ATCNN_model()
        self.model = model

class ECCNN(ElemNet):
    def __init__(self, name, number, save_model):
        super().__init__(name, number, save_model)
        self.model_name = 'ECCNN' + '_' + name + '_' + str(number)
        self.save_path = './models/' + self.model_name

    def build_model(self):
        model = ECCNN_model()
        self.model = model



if __name__ == '__main__':

    path = 'data/pervskite_data_roost.csv'
    data = pd.read_csv(path)



    # model.predict(False, path)
    import torch
