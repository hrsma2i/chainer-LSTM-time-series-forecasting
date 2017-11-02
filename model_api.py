#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import json
import re

import numpy as np
import pandas as pd
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
    serializers.load_npz(paht_weight, model)
    
    return model


# In[ ]:

root = 'result/test/adam0.1/'
pattern = r'model_epoch-[1-9]+'
[ l for l in os.listdir(result_path) if re.match(pattern, l)]



# In[ ]:

def predict(num_pred, model, prcsr, path_series_train):
    
    series = pd.read_csv(path_series_train, header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)
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

def predict(num_pred, model, prcsr, path_series_train):
    
    series = pd.read_csv(path_series_train, header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)
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

# fitting test
if __name__=="__main__":
    
    name_seq = 'airline'
    path_series_train = 'data/{}_train.csv'.format(name_seq)
    path_series_test  =  'data/{}_test.csv'.format(name_seq)
    
    prcsr = Processer()
    
    root = 'result/test/adam0.01'
    
    hp = json.load(open(os.path.join(root, 'hyperparameters.json')))
    units = hp['units']
    
    model = RNN(units)
    epoch = 65
    path_weight = os.path.join(root, 'model_epoch-{}'.format(epoch))
    serializers.load_npz(path_weight, model)
    
    pred_test = predict(num_pred=12, model=model, prcsr=prcsr,
                        path_series_train=path_series_train)
    obs_test = pd.read_csv(path_series_test, 
                           header=None).values.flatten()
    
    plt.figure(figsize=(20,10))
    plt.plot(pred_test)
    plt.plot(obs_test)


# In[ ]:



