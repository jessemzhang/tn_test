import numpy as np
import pickle
from ls_utils import *

datasets = [
    '10x',
    'Biase',
    'Birey',
    'Buettner',
    'Deng',
    'DropSeq',
    'inDrop',
    'Joost',
    'Kiselev',
    'Kolodziejczyk',
    'Patel',
    'Pollen',
    'Resolve',
    'Ting',
    'Treutlein',
    'Usoskin',
    'Yan',
    'Zeisel'
]

dirname = '/data/scRNASeq/datasets/'

d = {}

for dataset in datasets:

    loaded = load_dataset(dirname, dataset)
    if loaded is not None:
        X, Y = loaded

        print('Dataset %s loaded (%s cells, %s features).'%(dataset, len(X), len(X[0])))
        print('%s cell types'%(len(np.unique(Y))))
        for i in np.unique(Y):
            print('   %s cells of type %s'%(np.sum(Y == i), i))

        X, Y = preprocess(X, Y)
        d[dataset] = {}
        d[dataset]['train_accs'], d[dataset]['valid_accs'] = check_LR_classification(X, Y)

        print('-'*80)

with open('jz_script_output.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)