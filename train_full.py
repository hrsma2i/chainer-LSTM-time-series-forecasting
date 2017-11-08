#!/usr/bin/env python


# coding: utf-8

# In[ ]:

from data_process import Processer, name2prc
from train import train, name2
from model_api import select_hp


# In[ ]:

# train a model with train+val for comparison
if __name__=="__main__":
    data_root = 'data'
    root = 'result/test_full'
    name_seq = 'airline'
    name_prc = 'default'
    
    path_seq = os.path.join(root, name_seq)
    path_prc = os.path.join(path_seq, name_prc)
    name_hp = select_hp(root=path_prc, verbose=True)
    
    path_csv = os.path.join(data_root, '{}_train.csv'.format(name_seq))
    series = pd.read_csv(path_csv, header=None).values
        
    prcsr = Processer(**name2prc(name_prc))

    datasets = prcsr.get_datasets(series)
    
    # training
    #train(datasets, hp, out=root, n_epoch=300)

