from __future__ import print_function
import os, time, h5py
import numpy as np


# Load data
def load_mat_h5f(flname):
    h5f = h5py.File(flname,'r')
    X = h5f['dataset_1'][:]
    h5f.close()
    return X


def load_dataset(dirname, pref, get_features=False):
    
    X_path = os.path.join(dirname, pref, '%s_expr.txt'%(pref))
    Y_path = os.path.join(dirname, pref, '%s_labels.txt'%(pref))
    features_path = os.path.join(dirname, pref, '%s_features.txt'%(pref))

    if os.path.isfile(Y_path):
        Y = np.loadtxt(Y_path, dtype=str, delimiter='aSDFASDF')
    else:
        print('No labels found. Exiting.')
        return
    
    if os.path.isfile(X_path.replace('.txt', '.h5')):
        X = load_mat_h5f(X_path.replace('.txt', '.h5'))
    elif os.path.isfile(X_path):
        X = np.loadtxt(X_path).T
    else:
        print('No data found. Exiting.')
        return
       
    if get_features and os.path.isfile(features_path):
        features = np.loadtxt(features_path, dtype=str)
        return X, Y, features
    
    return X, Y


# See if labels are linearly separable

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

def preprocess(X, Y, folds=5, verbose=True):
    
    # Remove classes that have too few members
    keep_inds = np.ones(len(X)).astype(bool)
    for i in np.unique(Y):
        if np.sum(Y == i) < folds:
            keep_inds[Y == i] = False
    X, Y = X[keep_inds], Y[keep_inds]

    # Remove cells with low expression
    X = X[:, np.sum(X, 0) > 10]

    # Remove some genes (top 1000 based on dispersion)
    keep_inds = np.argsort(np.var(X, 0)/np.mean(X, 0))[-np.min((1000, len(X[0]))):]
    X = X[:, np.sort(keep_inds)]
    
    if verbose:
        print('%s cells, %s genes kept after preprocessing.'%np.shape(X))
    
    return X, Y


def check_LR_classification(X, Y):

    folds = 5
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

    train_accs = np.zeros(folds)
    valid_accs = np.zeros(folds)

    start = time.time()

    for i, (it, iv) in enumerate(splitter.split(X, Y)):
        Xt, Yt = X[it], Y[it]
        Xv, Yv = X[iv], Y[iv]

        model = LogisticRegression(multi_class='ovr')
        model.fit(Xt, Yt)
        Yt_hat = model.predict(Xt)
        Yv_hat = model.predict(Xv)

        train_accs[i] = np.sum(Yt == Yt_hat)/float(len(Yt))
        valid_accs[i] = np.sum(Yv == Yv_hat)/float(len(Yv))

        print('\r%s/%s folds done. Train acc: %.2f. Valid acc: %.2f. (%.2f s elapsed)' \
              %(i+1, folds, train_accs[i], valid_accs[i], time.time()-start),
                end='\n' if i == folds-1 else '')

    print('Mean train acc: %.2f +/- %.2f' \
          %(np.mean(train_accs), np.std(train_accs)/np.sqrt(folds)))
    print('Mean valid acc: %.2f +/- %.2f' \
          %(np.mean(valid_accs), np.std(valid_accs)/np.sqrt(folds)))
          
    return train_accs, valid_accs
