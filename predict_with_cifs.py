import argparse
import csv
from pathlib import Path
import functools
import json
import os
import random
import warnings
import math
import joblib

import pickle as pk
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import json
from model import CrystalGraphConvNet
from predict import ecsg_predict, ecsg_composition_predict


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]



class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, train=True,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.train = train
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        # random.seed(random_seed)
        # random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        if self.train:
            cif_id, target = self.id_prop_data[idx]
        else:
            cif_id = self.id_prop_data[idx][0]
            target = random.choice([True, False])

        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
           torch.stack(batch_target, dim=0), \
           batch_cif_ids


def composition_from_cif(cif_path):
    with open(cif_path+'/id_prop.csv') as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    formulas = []
    target = []
    for i in range(len(id_prop_data)):
        cif_file = id_prop_data[i][0]+ '.cif'
        cif_path_i = os.path.join(cif_path, cif_file)
        formulas.append(Structure.from_file(cif_path_i).composition.alphabetical_formula)
        target.append(random.choice([True, False]))

    id_prop_data = [eval(n[0]) for n in id_prop_data]
    df = pd.DataFrame(id_prop_data, columns=['materials-id'])
    df['composition'] = formulas
    df['target'] = target
    return df


def ecsg_struct_predict(cif_path, model_path, batch_size):
    data = CIFData(cif_path, train=False)
    cif_loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_pool)

    pre_test = []
    for i in range(5):
        for batch_idx, (input, target, _) in enumerate(cif_loader):
            x1, x2, x3, x4 = input

            model_path_i = os.path.join(model_path, f'CGCNN_fold_{i}.pth')
            state_dict = torch.load(model_path_i)
            model = CrystalGraphConvNet(orig_atom_fea_len=92, nbr_fea_len=41, classification=True)
            model.load_state_dict(state_dict)

            y = model(x1, x2, x3, x4)
            pre_y = np.exp(y.detach().numpy())[:, 1]

        pre_test.append(pre_y)

    return np.mean(pre_test, axis=0)


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Prediction script for structure-based model")

    # Add arguments to the parser
    parser.add_argument("--name", type=str, help="Name of the experiment or model", default='MP_cif_train_1')
    parser.add_argument("--cif_path", type=str, default='data/mp_2024_demo/cif',
                        help="Path to the dataset (default: data/mp_2024_demo/cif")
    parser.add_argument("--cgcnn_model_path", type=str, default='models',
                        help="Path to the dataset (default: models")

    parser.add_argument("--batchsize", type=int, default=2048,
                        help="Batch size for prediction (default: 2048)")
    parser.add_argument("--device", type=str, default='cuda:0',
                        help="Device to run the training on, e.g., 'cuda:0' or 'cpu', default: 'cuda:0'")
    args = parser.parse_args()

    name = args.name
    device = args.device
    cif_path = args.cif_path
    model_path = args.cgcnn_model_path
    batchsize = args.batchsize
    save_path = 'results/ + name + '_predict_results_cif.csv'

    df = composition_from_cif(cif_path)

    pre_composition_y = ecsg_composition_predict(name, df, folds=5)
    pre_structure_y = ecsg_struct_predict(cif_path, model_path, batch_size=64)
    meta_X = np.hstack((pre_composition_y, pre_structure_y.reshape(-1,1)))

    model = joblib.load(f'models/{name}_meta_model_structure.pkl')

    model.predict(meta_X)
    y = model.predict(meta_X)

    df = df.rename(columns={'target': 'pre_y'})
    df['pre_y'] = y
    results = [True if y[n] > 0.5 else False for n in range(len(y))]
    df['pre_y_01'] = results
    
    df.to_csv(save_path, index=False)
    print(f'Prediction results saved in {save_path}')




