#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import json

import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 10000)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from chainer import serializers

from data_process import Processer
from model import RNN


# In[ ]:

def get_learned_model(root, epoch):
    hp = json.load(open(os.path.join(root, 'hyperparameters.json')))
    units = hp['units']
    
    model = RNN(units)
    path_weight = os.path.join(root, 'model_epoch-{}'.format(epoch))
    serializers.load_npz(path_weight, model)
    
    return model


# In[ ]:

def select_epoch(root):
    """
    Select epoch in which the model have best val loss
    
    # Param
    - root (str): path where the model's weights each epoch are
    
    # Return
    - epoch (int): best epoch
    """
    path_log = os.path.join(root, 'log')
    df_log = pd.read_json(path_log)
    
    best_idx = df_log['validation/main/loss'].argmin()
    epoch = int(df_log['epoch'].ix[best_idx])
    return epoch


# In[ ]:

def select_hp(root, verbose=False):
    """
    Select hyperparameter with which 
    the model have best mean of best-10 val loss
    
    # Param
    - root (str): path where the model's weights each epoch are
    
    # Return
    - best_hp (str): best hyperparameters dir name
    """
    best_n = 10
    key = 'validation/main/loss'

    names_hp = (b.name for b in os.scandir(root) if b.is_dir())

    hp_score = []

    for name_hp in names_hp:
        path_hp = os.path.join(root, name_hp)
        path_log = os.path.join(path_hp, 'log')
        df_log = pd.read_json(path_log)
        
        scores = df_log[key].sort_values().values
        eval_score = np.mean(scores[:best_n])
        hp_score.append((name_hp, eval_score))

    hp_score = pd.DataFrame(hp_score,
                           columns=['hyperparamter', key])
    hp_score = hp_score.sort_values(key)
    hp_score = hp_score.reset_index(drop=True)
    best_hp = hp_score.ix[0, 0]
    
    if verbose:
        display(hp_score)
    
    return best_hp


# In[ ]:

def predict(num_pred, model, prcsr, path_csv_train):
    
    series = pd.read_csv(path_series_train, header=None).values
    X_train, X_val, _, _ = prcsr.transform_train(series)
    X_train = np.concatenate((X_train, X_val), axis=0)
    
    # setup hidden state for predicting test
    model.reset_state()
    for Xt in X_train:
        _ = model(Xt.reshape(-1, 1)).data[0]
    
    # make prediction
    pred = []
    p_t = X_train[-1]
    for _ in range(num_pred):
        p_t = model(p_t.reshape(-1, 1)).data[0]
        pred.append(p_t)
    pred = np.array(pred)
    
    if prcsr.ysclr is not None:
        pred = prcsr.inverse_scale(pred)

    if prcsr.diff:
        pred_diff = pred.copy()
        pred = []
        p_t = prcsr.last_before_diff
        for d_t in pred_diff:
            p_t += d_t
            pred.append(p_t.copy())
        pred = np.array(pred)
    
    if prcsr.log:
        pred = np.expm1(pred)
        
    return pred


# In[ ]:

# fitting test
if __name__=="__main__":
    data_root = 'data'
    root      = 'result/test'
    name_seq  = 'airline'
    name_prc  = 'default'
    
    name_csv_train = '{}_train.csv'.format(name_seq)
    path_csv_train = os.path.join(data_root, name_csv_train)
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    
    prcsr = Processer()
    
    path_seq = os.path.join(root, name_seq)
    path_prc = os.path.join(path_seq, name_prc)
    
    name_hp = select_hp(root=path_prc)
    
    path_hp = os.path.join(path_prc, name_hp)
    epoch = select_epoch(root=path_hp)
    
    model = get_learned_model(root=path_hp, epoch=epoch)
    
    pred_test = predict(num_pred=12, model=model, prcsr=prcsr,
                        path_csv_train=path_csv_train)
    obs_test = pd.read_csv(path_series_test, 
                           header=None).values.flatten()
    
    print(name_hp)
    print(epoch)
    plt.figure(figsize=(20,10))
    plt.plot(pred_test, label='pred')
    plt.plot(obs_test,  label='obs')
    plt.legend()


# In[ ]:

# fitting train val
if __name__=="__main__":
    
    prcsr = Processer()
    
    root = 'result/test/adam0.1'
    epoch = 300
    
    hp = json.load(open(os.path.join(root, 'hyperparameters.json')))
    units = hp['units']
    
    model = RNN(units)
    serializers.load_npz(os.path.join(root, 'model_epoch-{}'.format(epoch)), model)
    
    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)
        
    X_train, X_val, y_train, y_val = prcsr.transform_train(series)
    
    X_train = np.concatenate((X_train, X_val), axis=0)
    obs_train = np.concatenate((y_train, y_val), axis=0)
    
    pred_train = []
    
    model.reset_state()
    for Xt in X_train:
        pred = model(Xt.reshape(-1, 1)).data[0]
        pred_train.append(pred)
    pred_train = np.array(pred_train)
    
    plt.figure(figsize=(20,10))
    plt.axvline(y_train.shape[0], color='red')
    plt.plot( obs_train)
    plt.plot(pred_train)
    
    
    if prcsr.ysclr is not None:
        obs_train = prcsr.inverse_scale(obs_train)
        pred_train = prcsr.inverse_scale(pred_train)
        
        plt.figure(figsize=(20,10))
        plt.axvline(y_train.shape[0], color='red')
        plt.plot( obs_train)
        plt.plot(pred_train)
        
    if prcsr.diff:
        before_diff = series[:,:]
        if prcsr.log:
            before_diff = prcsr.log_transform(before_diff)
        obs_train  = before_diff[2:] 
        obs1_train = before_diff[1:-1]
        pred_train = prcsr.inverse_diff_given(pred_train, obs1_train)
        
        plt.figure(figsize=(20,10))
        plt.axvline(y_train.shape[0], color='red')
        plt.plot( obs_train)
        plt.plot(pred_train)
    
    if prcsr.log:
        pred_train = prcsr.inverse_log(pred_train)
        
    obs_train = series[2:]
    
    plt.figure(figsize=(20,10))
    plt.axvline(y_train.shape[0], color='red')
    plt.plot( obs_train)
    plt.plot(pred_train)


# In[ ]:



