#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split


# In[ ]:

# TODO: validation 時に、MSE以外のscoreもlogに残せるように！
#       train中にscore計算できないようなerrorが生じたら、skipするように

class Processer(object):
    """
    # Attr
    - log   (bool): log or not
    - diff  (bool): difference or not
    - sclr  (Scaler): scaler for input
    - ysclr (Scaler): scaler for pred or label
    - last_before_diff (int/float): last data of diffed series
                                    to compute pred before diffed
    """
    
    def __init__(self, log=True, diff=True, 
                 sclr=MinMaxScaler((-1,1)), ysclr=MinMaxScaler((-1,1))):
        self.log   = log
        self.diff  = diff
        self.sclr  = sclr
        self.ysclr = ysclr
        
        
    def get_datasets(self, series):
        """
        # Param
        - series (ndarray: (T, F)): raw time series
        T: time steps
        
        # Return
        - ds_train (list): list(zip(X_train, y_train)) 
            instead of Chainer's Dataset obj
        - ds_val   (list) 
        
        # each sequence explanation
        - X_train (ndarray: (S, T, F))
        - X_val   (ndarray: (S, T, F))
        - y_train (ndarray: (S, T, F))
        - y_val   (ndarray: (S, T, F))
        S: samples = 1
        T: time steps
        F: features = 1
        
        """
        
        X_train, X_val, y_train, y_val = self.transform_train(series)
        # change type
        X_train = X_train.astype(np.float32)
        X_val   =   X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val   =   y_val.astype(np.float32)
        # change shape
        X_train = X_train[np.newaxis, :, :]
        X_val   =   X_val[np.newaxis, :, :]
        y_train = y_train[np.newaxis, :, :]
        y_val   =   y_val[np.newaxis, :, :]
        ds_train = list(zip(X_train, y_train))
        ds_val   = list(zip(X_val  , y_val  ))
        
        return ds_train, ds_val
        
        
    def transform_train(self, series):
        """
        # Param
        - series  (ndarray: (T, F)): raw time series
        T: time steps
        F: features
        
        # Return
        - X_train (ndarray: (T, F))
        - X_val   (ndarray: (T, F))
        - y_train (ndarray: (T, F))
        - y_val   (ndarray: (T, F))
        S: samples = 1
        T: time steps
        F: features
        
        # Flow
        - (log)
        - (difference)
        - supervise
        - train val split
        - (scale)
        - change shape for RNN
        """
        series = series.astype(np.float32)
        self.last_raw = series[-1]
        
        if self.log:
            series = self.log_transform(series)
        
        if self.diff:
            self.last_before_diff = series[-1]
            series = self.difference(series)
            
        X, y = self.supervise(series)
        
        Xtr_Xvl_ytr_yvl = self.train_val_split(X, y)
        
        if self.sclr is not None:
            Xtr_Xvl_ytr_yvl = self.scale(*Xtr_Xvl_ytr_yvl)
        
        return Xtr_Xvl_ytr_yvl
    
    
    def log_transform(self, series):
        """
        # Param
        - series (ndarray: (T, F)): raw series
        
        # Return
        -  (ndarray: (T, F)): logged series
        """
        return np.log1p(series)
    
    def inverse_log(self, pred):
        return np.expm1(pred)
            
    
    def difference(self, series):
        """
        # Param
        - seires (ndarray: (T,   F))
        
        # Return
        - diffed (ndarray: (T-1, F)): diffed series
        """
        diffed =  series[1:] - series[:-1]
        
        return diffed
    
    def inverse_diff_given(self, pred, obs1):
        """
        # Param
        - pred (ndarray: (T, 1))
        - obs1 (ndarray: (T, 1)): the previous time observation
        
        # Return
        - (ndarray: (T, 1))
        """
        
        return pred + obs1
    
    def supervise(self, series):
        """
        # Param
        - series (ndarray: (T, F))
        
        # Return
        - X (ndarray: (T-1, F)): input
        - y (ndarray: (T-1, F)): label
        """
        X = series[:-1]
        y = series[1:]
        return X, y
    
    def train_val_split(self, X, y):
        """
        # Param
        - X (ndarray: (T, F)): input
        - y (ndarray: (T, F)): label
        
        # Return
        - X_train, X_val, y_train, y_val 
            (ndarray: (T_train/val, F))
        """
        val_size = 0.3
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                         test_size=val_size, shuffle=False)
        return X_train, X_val, y_train, y_val
    
    def scale(self, X_train, X_val, y_train, y_val):
        """
        # Param
        - X_train, X_val, y_train, y_val 
            (ndarray: (T, F))
        
        # Return
        - X_train, X_val, y_train, y_val 
            (ndarray: (T, F)): scaled
        """
        
        X_train = X_train.astype(np.float32)
        X_val   =   X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val   =   y_val.astype(np.float32)
        
        X_train = self.sclr.fit_transform(X_train)
        X_val   = self.sclr.transform(X_val)
        
        if self.ysclr is not None:
            y_train = self.ysclr.fit_transform(y_train)
            y_val   = self.ysclr.transform(y_val)
            
        return X_train, X_val, y_train, y_val
    
    def inverse_scale(self, pred):
        """
        # Param
        - pred (ndarray: (T, 1))
        
        # Return
        - (ndarray: (T, 1))
        """
        return self.ysclr.inverse_transform(pred)


# In[ ]:

def test_pre_prcsr(log, diff, sclr, ysclr):
    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)
    #series = series[:102]
    print('raw', series.shape)
    print(series[:5])
    print()
    plt.plot(series)
    plt.show()
    
    prcsr = Processer(log=log, diff=diff, 
                      sclr=sclr, ysclr=ysclr)
    
    # log
    if prcsr.log:
        series = prcsr.log_transform(series)
        print('logged', series.shape)
        print(series[:5])
        print()
        plt.plot(series)
        plt.show()
    
    
    # diff
    if prcsr.diff:
        series = prcsr.difference(series)
        print('diff', series.shape)
        print(series[:5])
        print()
        plt.plot(series)
        plt.show()
        
    # supervise
    print('supervise')
    X, y = prcsr.supervise(series)
    print('X', X.shape)
    print(X[:5])
    print('y', y.shape)
    print(y[:5])
    print()
    
    # train val split
    print('train val split')
    Xtr_Xvl_ytr_yvl = prcsr.train_val_split(X, y)
    names = ('X_train', 'X_val', 'y_train', 'y_val')
    for name, seq in zip(names, Xtr_Xvl_ytr_yvl):
        print(name, seq.shape)
        print(seq[:5])
    print()
        
    # scale
    if prcsr.sclr is not None:
        print('scale')
        Xtr_Xvl_ytr_yvl = prcsr.scale(*Xtr_Xvl_ytr_yvl)
        names = ('X_train_scl', 'X_val_scl', 'y_train_scl', 'y_val_scl')
        scores = ('min', 'max', 'mean', 'std')
        for name, seq in zip(names, Xtr_Xvl_ytr_yvl):
            print(name, seq.shape)
            print(seq[:5])
            print(pd.Series(seq.flatten()).describe()[['min', 'max', 'mean', 'std']])
        print()
        
    # datasets for RNN
    ds_train, ds_val = prcsr.get_datasets(series)
    print('train samples', len(ds_train))
    print('input (T, F)')
    print(ds_train[0][0].shape)
    print('label (T, F)')
    print(ds_train[0][1].shape)
    print()
    print('val samples', len(ds_val))
    print('input (T, F)')
    print(ds_val[0][0].shape)
    print('label (T, F)')
    print(ds_val[0][1].shape)
    print()


# In[ ]:

# fitting
if __name__=="__main__":
    import json
    
    from chainer import serializers
    
    from model import RNN
    
    
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

def predict(num_pred, model, prcsr, path_series_train):
    
    series = pd.read_csv(path_series_train, header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)
    X_train, X_val, _, _ = prcsr.transform_train(series)
    X_train = np.concatenate((X_train, X_val), axis=0)
    
    model.reset_state()
    # setup hidden state for predicting test
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

# fitting
if __name__=="__main__":
    import json
    
    from chainer import serializers
    
    from model import RNN
    
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

# preprocess
if __name__=="__main__":
    # configs
    configs = {
        'log':True,
        'diff':True,
        'sclr':MinMaxScaler(feature_range=(-1,1)),
        'ysclr':MinMaxScaler(feature_range=(-1,1)),
    } 
    test_pre_prcsr(**configs)


# In[ ]:

def loop_prc(func):
    search_space={
        'log':[True, False],
        'diff':[True, False],
        'sclr':['MinMax', 'MinMax+', 'Standard', None],
        'ysclr':[True, False]
    }


# In[ ]:



