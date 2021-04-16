from torch.utils.data import Dataset

import pickle
import numpy as np
import os
import torch 
import h5py

from random import randint

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, videos_pkl, in_file, m_file):
        super(dataset_h5, self).__init__()
        f2 = open(videos_pkl,"r")
        videos = f2.readlines()
        self.__avid__=[]
        self.__nvid__=[]
        for v in videos:
            if 'var' in videos_pkl:
                if 'Normal' in v.strip().split('/')[-1][:-4]:
                    self.__nvid__.append(v.strip().split('/')[-1].split(' ')[0])
                else:
                    self.__avid__.append(v.strip().split('/')[-1].split(' ')[0])
            else:
                if 'Normal' in v.strip().split('/')[-1][:-4]:
                    self.__nvid__.append(v.strip().split('/')[-1][:-4])
                else:
                    self.__avid__.append(v.strip().split('/')[-1][:-4])
        self.file = h5py.File(in_file, 'r')
        self.mask_file = h5py.File(m_file, 'r')

    def __getitem__(self, index):
        nvid = self.__nvid__[index]
        ind = (index+randint(0, 809))%810
        avid = self.__avid__[ind]
        
        feas = []
        preds = []
        from sklearn.preprocessing import Normalizer
        tmp = self.file[avid]
        if tmp.shape[0]%32 ==0:
            fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
        else:
            feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
            add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]
            add_fea = np.tile(feat[-1],(add, 1))
            fea_new = np.concatenate((feat, add_fea), axis=0)
            fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))

        ano_fea = fea_new.mean(axis=1)
        # ano_fea = Normalizer(norm='l2').fit_transform(ano_fea)
        preds.append(1)

        tmp = self.file[nvid]
        if tmp.shape[0]%32 ==0:
            fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
        else:
            feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
            add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]
            add_fea = np.tile(feat[-1],(add, 1))
            fea_new = np.concatenate((feat, add_fea), axis=0)
            fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))
        nor_fea = fea_new.mean(axis=1)

        preds.append(0)

        ano_feas = torch.from_numpy(ano_fea)
        nor_feas = torch.from_numpy(nor_fea)

        preds = torch.Tensor(preds)

        return ano_feas, nor_feas, preds

    def __len__(self):
        return min(len(self.__avid__),len(self.__nvid__))

class dataset_h5_test(torch.utils.data.Dataset):
    def __init__(self, videos_pkl, in_file):
        super(dataset_h5_test, self).__init__()
        f2 = open(videos_pkl,"r")
        videos = f2.readlines()
        self.__vid__=[]

        for v in videos:
            self.__vid__.append(v.strip().split('/')[-1].split(' ')[0])
        self.file = h5py.File(in_file, 'r')
        # import pdb;pdb.set_trace()
        

    def __getitem__(self, index):
        vid = self.__vid__[index]
 
        # import pdb;pdb.set_trace()
        
        feas = []
        preds = []
        from sklearn.preprocessing import Normalizer
        tmp = self.file[vid]
        if tmp.shape[0]%32 ==0:
            fea_new = np.reshape(tmp, (32,-1, tmp.shape[1]))
        else:
            feat = np.resize(tmp, (tmp.shape[0], tmp.shape[1]))
            add =  (tmp.shape[0]//32+1) * 32 - tmp.shape[0]
            add_fea = np.tile(feat[-1],(add, 1))
            fea_new = np.concatenate((feat, add_fea), axis=0)
            fea_new = np.reshape(fea_new, (32,-1, tmp.shape[1]))

        ano_fea = fea_new.mean(axis=1)
        # if 'Explosion010_x264' in vid:
        #     import pdb;pdb.set_trace()
        # ano_feas = torch.from_numpy(ano_fea)
        # ano_fea = Normalizer(norm='l2').fit_transform(ano_fea)
        pred = 0
        if 'Normal' not in vid:
            pred += 1
        return ano_fea, pred, vid

    def __len__(self):
        return len(self.__vid__)
