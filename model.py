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

    def predict(self, df, cuda, device, n_fold):
        model_name = self.model_name
        df_new = df.reset_index(drop=True)
        model_5_predict(df_new, cuda, device,  n_fold, model_name)
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



class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


# class Postion_encoder(nn.Module):
#
#     def __init__(self,src):
#         super(Postion_encoder, self).__init__()
#         self

class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.
        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        # print(atom_fea.shape) shape = (N, 64)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features
        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch
        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


if __name__ == '__main__':

    path = 'data/pervskite_data_roost.csv'
    data = pd.read_csv(path)



    # model.predict(False, path)
    import torch
