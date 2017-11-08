#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_process import Processer, name2prc
from train import tune


# In[ ]:

def tune_prc(root, series, n_sample=100, n_epoch=300):
    """
    # Param
    - root   (str): the path where results will be saved
    - series (ndarray: (T, F)):
    """
    def routine(name_prc):
        """
        global
        - root
        - series
        - n_sample
        - n_epoch
        """
        print(name_prc)
        prcsr = Processer(name2prc(name_prc))
        path_prc = os.path.join(root, name_prc)
        if not os.path.exists(path_prc):
            os.mkdir(path_prc)
        datasets = prcsr.get_datasets(series)
        tune(root=path_prc, datasets=datasets,
            n_sample=n_sample, n_epoch=n_epoch)
        
    prcs = [
        'not_log',
        'not_diff',
        'minmax+',
        'standard',
        'not_scale',
        'not_label_scale',
    ]
    
    for name_prc in prcs:
        routine(name_prc)


# In[ ]:

def tune_seq(data_root, path_sequences, root, 
            n_sample=100, n_epoch=300):
    """
    # Param
    - data_root (str): where seq csv-data are
    - path_sequences (str): where sequences-list are
    - root (srt): root dir where results will be dumped
    """
    seqs = [ seq.rstrip()
            for seq in open(path_sequences, 'r').readlines()]
    
    for name_seq in seqs:
        print(name_seq)
        path_csv = os.path.join(data_root, 
                                '{}_train.csv'.format(name_seq))

        series = pd.read_csv(path_csv, header=None).values

        path_seq = os.path.join(root, name_seq)
        if not os.path.exists(path_seq):
            os.mkdir(path_seq)

        tune_prc(root=path_seq, series=series, 
                 n_sample=n_sample, n_epoch=n_epoch)


# In[ ]:

if __name__=="__main__":
    # for test
    #n_sample = 2
    #n_epoch  = 2
    n_sample = 100
    n_epoch  = 300
    data_root = 'data'
    path_sequences = os.path.join(data_root, 'sequences_verify')
    root = 'result/test_tune_prc'
    if not os.path.exists(root):
        os.mkdir(root)
    tune_seq(data_root=data_root,
            path_sequences=path_sequences, root=root,
            n_sample=n_sample, n_epoch=n_epoch)


# In[ ]:



