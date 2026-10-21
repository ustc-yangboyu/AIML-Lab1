import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np
import pandas as pd
import sklearn

import src.utils as utils

def diabetes():
    data = sklearn.datasets.load_diabetes()
    return {'X': data.data, 'Y': data.target, 'Name': data.feature_names}

def split_datasets(X, Y, test_size = 0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    n_test = int(X.shape[0] * test_size)
    return {
        "X_train": X[indices[n_test:]],
        "Y_train": Y[indices[n_test:]],
        "X_test": X[indices[:n_test]],
        "Y_test": Y[indices[:n_test]],
    }

def load_features(data_dir, stem, layer):
    for sub in ('layers_all', 'processed', '.'):
        path = Path(data_dir) / sub / f'{stem}.layer{layer}.npy'
        if path.exists():
            return np.load(path).astype(np.float64)
    raise FileNotFoundError(f'{stem}.layer{layer}.npy not found under {data_dir}')

WORLD_DIRS = {'base': 'data/world_base', 'extra': 'data/world_extra'}

def load_world(version = 'base', layer = 11, use_exist = True):
    data_dir = Path(WORLD_DIRS.get(version, version))
    for split in ('train', 'valid'):
        if not use_exist or not (data_dir / 'processed' / f'{split}.layer{layer}.npy').exists():
            utils.extract_layers(
                data_dir / 'origin' / f'{split}.csv', 
                data_dir / 'processed', 
                layers = [layer], 
                token_selection = 'country'
            )

    train_csv = pd.read_csv(data_dir / 'origin' / 'train.csv')
    valid_csv = pd.read_csv(data_dir / 'origin' / 'valid.csv')
    return {
        'X_train': load_features(data_dir, 'train', layer),
        'Y_train': train_csv[['latitude', 'longitude']].to_numpy(np.float64),
        'X_valid': load_features(data_dir, 'valid', layer),
        'Y_valid': valid_csv[['latitude', 'longitude']].to_numpy(np.float64),
        'Train_Countries': train_csv.country,
        'Valid_Countries': valid_csv.country,
    }

def load_truth(data_dir = 'data/truth', layer = 23):
    train_csv = pd.read_csv(Path(data_dir) / 'origin' / 'train.csv')
    valid_csv = pd.read_csv(Path(data_dir) / 'origin' / 'valid.csv')
    return {
        'X_train': load_features(data_dir, 'train', layer),
        'Y_train': train_csv.label.to_numpy(np.float64),
        'X_valid': load_features(data_dir, 'valid', layer),
        'Y_valid': valid_csv.label.to_numpy(np.float64),
    }

def load_world_submission(results_dir = 'outputs/results'):
    with open(Path(results_dir) / 'Q5_final_config.json') as f:
        Final = json.load(f)
    data = load_world(Final['version'], Final['layer'])
    with np.load(Path(results_dir) / 'world.npz') as z:
        Y_pred = data['X_valid'] @ z['weights'].T + z['bias']
    return Final, data, Y_pred
