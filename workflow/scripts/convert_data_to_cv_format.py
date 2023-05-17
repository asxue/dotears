import os
import argparse
from sklearn.model_selection import KFold
import numpy as np

"""
Takes a single .npz data file and splits into k (train, validation) folds for cross validation.
Folds are written to separate files, whose parent directory is specified by args.
"""


def split_interventional_data_cv(data, folds, random_state):
    train_splits = [({}, {}) for _ in range(folds)]
    kf = KFold(n_splits=folds, random_state=random_state, shuffle=True)

    for key, arr in data.items():
        for split, (train_cv_index, val_cv_index) in enumerate(kf.split(arr)):
            train_splits[split][0][key] = arr[train_cv_index]
            train_splits[split][1][key] = arr[val_cv_index]

    return train_splits

if __name__ == '__main__':
    """
    args:
        data: path to initial .npz file
        out_folder: parent directory for k-fold split .npz files
        folds: number of folds, k, for cross-validation
        random_state: random state passed to sklearn for partitioning
        sim: sim number 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='data file', type=str)
    parser.add_argument('--out_folder', help='folder to output fold data', type=str)
    parser.add_argument('--folds', help='number of folds for cross validation', type=int)
    parser.add_argument('--random_state', help='random state for k fold split', type=int)
    parser.add_argument('--sim', type=int, help='simulation number')
    args = parser.parse_args()

    data = np.load(args.data)

    train_splits = split_interventional_data_cv(data, args.folds, args.random_state)    
    for i, x in enumerate(train_splits):
        train_data, val_data = x

        train_path = os.path.join(args.out_folder, 'fold{}_train.npz'.format(str(i)))
        val_path = os.path.join(args.out_folder, 'fold{}_val.npz'.format(str(i)))
        
        if not os.path.exists(args.out_folder):
            os.makedirs(args.out_folder)
        np.savez(train_path, **train_data)
        np.savez(val_path, **val_data)
