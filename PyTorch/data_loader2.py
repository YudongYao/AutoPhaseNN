# load complex diff, then ifft to get ground truth obj for testing
from typing import Text, TextIO
import json
import numpy as np
import os
import torch
import random


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_ID, data_path, ratio=0.9, dataset='all', load_all=False, scale_I=0, shuffle=True):
        'Initialization'
        if dataset == 'all':
            self.data_ID = data_ID
        else:
            n = int(len(data_ID) * ratio) # ratio of training data 
            pos = list(range(len(data_ID)))
            if shuffle:
                # give a random seed, so the training and validation data are not overlapped
                random.Random(4).shuffle(pos)
            if dataset == 'train':
                self.data_ID = [data_ID[k] for k in pos[:n]]
            elif dataset == 'validation':
                self.data_ID = [data_ID[k] for k in pos[n:]]
            elif dataset=='test':
                self.data_ID = [data_ID[k] for k in pos]
            else:
                raise AssertionError("Unexpected value of dataset name!", dataset)
                
        self.data_path = data_path
        self.load_all = load_all
        self.scale_I = scale_I
        if self.load_all:
            """
                if load all data, initialize the database
                load all data will take a lot of memory.
            """
            data_folder = self.data_path
            diff_list = []
            amp_list = []
            phi_list = []
            for img_n in self.data_ID:

#                 diff = np.load(self.data_path+img_n)['arr_0']
#                 realspace = np.load(self.data_path+img_n)['arr_1']
                data = np.load(self.data_path+img_n)
                realspace = np.fft.ifftn(np.fft.ifftshift(data))
                
                diff = np.abs(data)
                amp = np.abs(realspace)
                phi = np.angle(realspace)
                
                if self.scale_I>0:
                    max_I = diff.max()
                    diff = diff/(max_I+1e-6)*self.scale_I

                diff_list.append(diff[np.newaxis])
                amp_list.append(amp[np.newaxis])
                phi_list.append(phi[np.newaxis])
            
            self.diff_list = diff_list
            self.amp_list = amp_list
            self.phi_list = phi_list

            print('All data loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_ID)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.load_all:
            return np.array(self.diff_list[index]),\
                   np.array(self.amp_list[index]), \
                   np.array(self.phi_list[index])
        else:
            # Select sample
            img_ID = self.data_ID[index]
            data_folder = self.data_path
#             diff = np.load(self.data_path+img_ID)['arr_0']
#             realspace = np.load(self.data_path+img_ID)['arr_1']
            data = np.load(self.data_path+img_ID)
            realspace = np.fft.ifftn(np.fft.ifftshift(data))
        
            diff = np.abs(data).astype('float32')
            amp = np.abs(realspace).astype('float32')
            phi = np.angle(realspace).astype('float32')

            if self.scale_I>0:
                max_I = diff.max()
                diff = diff/(max_I+1e-6)*self.scale_I


            return diff[np.newaxis], amp[np.newaxis], phi[np.newaxis]
