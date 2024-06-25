import argparse
import pickle
import pandas as pd

from utils.feature_engineering import Magpie_fea, ECCNN_fea



def featurization(formulas ):

    f_0 = Magpie_fea(formulas)
    f_1 = []
    f_2 = ECCNN_fea(formulas)

    features = [f_0, f_1, f_2]

    return features


def save_feature(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)
    print(f'save features to {path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default='data/datasets/demo_mp_data.csv')
    parser.add_argument("--feature_path", type=str, default='data/datasets/demo_feature')

    args = parser.parse_args()
    data_path = args.path
    save_path = args.feature_path

    data_comp = pd.read_csv(data_path)['composition'].values
    features = featurization(data_comp)
    save_feature(features, save_path)
