import os
import os.path

import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class2lbl = {
    'blues'     : 0, 
    'classical' : 1, 
    'country'   : 2, 
    'disco'     : 3, 
    'hiphop'    : 4, 
    'jazz'      : 5,
    'metal'     : 6, 
    'pop'       : 7, 
    'reggae'    : 8
}

def default_flist_reader(flist):
    df_mfccs = pd.read_csv(flist, header=None)
    return df_mfccs

class MfccLoaderCNN(data.Dataset):
    def __init__(self, root, flist, means, stds, flist_reader=default_flist_reader):
        self.mfccs_list = flist_reader(flist)
        self.means = means
        self.stds = stds
        self.root = root

    def __getitem__(self, index):
        filename = self.mfccs_list.loc[index, 1]
        target = class2lbl[filename.split('/')[0]]
        filename = os.path.join(self.root, filename + '.npy')
        mfcc = np.load(filename).T
        index = np.random.randint(0, mfcc.shape[0] - 191)
        mfcc = mfcc[index:index+190].reshape(1, 190, mfcc.shape[1])
        mfcc = np.array(mfcc, dtype=float)
        if self.means:
            mfcc = mfcc - self.means
        if self.stds:
            mfcc = mfcc / self.stds
        return torch.from_numpy(mfcc).float(), target - 1

    def __len__(self):
        return len(self.mfccs_list)
