#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os

import numpy as np
import pandas as pd

from data_process import Processer, name2prc
from train import train, path_hp2hp
from model_api import select_hp


# In[ ]:

def train_full(data_root='data',
               root='result/test',
               name_seq='airline',
               name_prc='default',
               n_epoch=150):
    path_seq = os.path.join(root, name_seq)
    path_prc = os.path.join(path_seq, name_prc)
    name_hp = select_hp(root=path_prc, verbose=True)
    path_hp = os.path.join(path_prc, name_hp)
    
    path_csv = os.path.join(data_root, '{}_train.csv'.format(name_seq))
    series = pd.read_csv(path_csv, header=None).values
    #val_size = 0.3
    size_pad = round(series.shape[0] * (3./7))
    z = np.zeros((size_pad, series.shape[1]))
    series_pad = np.concatenate((series, z), axis=0)
    print('series_pad', series_pad.shape)
    print()
    
        
    prcsr = Processer(**name2prc(name_prc))

    datasets = prcsr.get_datasets(series_pad)
    X_train, _, _, _ = prcsr.transform_train(series_pad)
    print('REMARK: X_train is fewer because diff and supervise')
    print('sereis', series.shape)
    print('X_train', X_train.shape)
    print()
    
    hp = path_hp2hp(path_hp)
    
    out = os.path.join(path_prc, 'full_'+name_hp)
    print(out)
    
    # training
    train(datasets, hp, out=out, n_epoch=n_epoch)


# In[ ]:

# test train_full
if __name__=="__main__":
    data_root = 'data'
    root = 'result/test'
    name_prc = 'default'
    
    name_seq = 'toy'
    
    train_full(data_root=data_root,
               root=root,
               name_seq=name_seq,
               name_prc=name_prc)


# In[ ]:

# loop train_full over seqs
if __name__=="__main__":
    data_root = 'data'
    root = 'result/test'
    name_prc = 'default'
    name_sequences = 'sequences'
    
    path_sequences = os.path.join(data_root, name_sequences)
    seqs = [ seq.rstrip()
            for seq in open(path_sequences, 'r').readlines()] 
    
    for name_seq in seqs:
        train_full(data_root=data_root,
                   root=root,
                   name_seq=name_seq,
                   name_prc=name_prc)


# In[ ]:



