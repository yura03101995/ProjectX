import torch.utils.data as data
import torch
import os
import pandas as pd
import numpy as np
import os.path

def default_flist_reader(flist):
    df_mfccs = pd.read_csv(flist)
    return df_mfccs

class MfccLoader(data.Dataset):
    def __init__(self, flist, means, stds, root=None, flist_reader=default_flist_reader):
        self.mfccs_list = flist_reader(flist)
        self.means = means
        self.stds = stds

    def __getitem__(self, index):
        mfcc = self.mfccs_list.loc[index, ['mfcc_' + str(i) for i in range(1, 40)]] 
        target = self.mfccs_list.loc[index, 'lbl']
        mfcc = np.array(mfcc, dtype=float)
        if self.means:
            mfcc = mfcc - self.means
        if self.stds:
            mfcc = mfcc / self.stds
        return torch.from_numpy(mfcc).float(), target - 1

    def __len__(self):
        return len(self.mfccs_list)
