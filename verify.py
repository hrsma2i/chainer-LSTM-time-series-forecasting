#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_process import Processer
from train import tune


# In[ ]:

def tune_prc(root, series, n_sample=100, n_epoch=300):
    """
    # Param
    - root   (str): the path where results will be saved
    - series (ndarray: (T, F)):
    """
    def routin(name_prc, prscr):
        """
        global
        - root
        - series
        - n_sample
        - n_epoch
        """
        print(name_prc)
        path_prc = os.path.join(root, name_prc)
        if not os.path.exists(path_prc):
            os.mkdir(path_prc)
        datasets = prcsr.get_datasets(series)
        tune(root=path_prc, datasets=datasets,
            n_sample=n_sample, n_epoch=n_epoch)
        
    name_prc = 'not_log'
    prcsr = Processer(log=False)
    routin(name_prc, prcsr)
        
    name_prc = 'not_diff'
    prcsr = Processer(diff=False)
    routin(name_prc, prcsr)
    
    sclrs ={
        'minmax':MinMaxScaler((-1,1)),
        'minmax+':MinMaxScaler((0,1)),
        'standard':StandardScaler(),
        'not_scale':None,
    }
    for name_prc, sclr in sclrs.items():
        prcsr = Processer(sclr=sclr, ysclr=sclr)
        routin(name_prc, prcsr)
        
    name_prc = 'not_label_scale'
    prcsr = Processer(ysclr=None)
    routin(name_prc, prcsr)


# In[ ]:

data_root = 'data'
name_seq = 'airline'

for name_seq in f.readlines:

path_csv = os.path.join(data_root, 
                        '{}_train.csv'.format(name_seq))

series = pd.read_csv(path_csv, header=None).values

root = 'result/test_prc_loop'
if not os.path.exists(root):
    os.mkdir(root)
    
tune_prc(root=path_seq, series=series, 
         n_sample=2, n_epoch=2)


# In[ ]:



