#!/usr/bin/env python


# coding: utf-8

# In[ ]:

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
    - log_trnsfmr (FanctionTransformer)
    - diff  (bool): difference or not
    - sclr  (Scaler): scaler for input
    - ysclr (Scaler): scaler for pred or label
    - last_raw  (int/float): last data of raw series
                            to compute pred raw
    - last_diff (int/float): last data of diffed series
                            to compute pred diffed
    """
    
    def __init__(self, log_trnsfmr=FunctionTransformer(np.log1p),
                 diff=True, 
                 sclr=MinMaxScaler(), ysclr=MinMaxScaler()):
        self.log_trnsfmr   = log_trnsfmr
        self.diff  = diff
        self.sclr  = sclr
        self.ysclr = ysclr
        
        
    def get_datasets(self, series):
        """
        # Param
        - series (ndarray: (T, )): raw time series
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
        X_train = X_train.astype(np.float32)
        X_val   =   X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val   =   y_val.astype(np.float32)
        ds_train = list(zip(X_train, y_train))
        ds_val   = list(zip(X_val  , y_val  ))
        
        return ds_train, ds_val
        
        
    def transform_train(self, series):
        """
        # Param
        - series (ndarray: (T, )): raw time series
        T: time steps
        
        # Return
        - X_train (ndarray: (T, 1))
        - X_val   (ndarray: (T, 1))
        - y_train (ndarray: (T, 1))
        - y_val   (ndarray: (T, 1))
        S: samples = 1
        T: time steps
        F: features = 1
        
        # Flow
        - (difference)
        - supervise
        - train val split
        - (scale)
        - change shape for RNN
        """
        series = series.astype(np.float32)
        self.last_raw = series[-1]
        
        if self.diff:
            series = self.difference(series)
            self.last_diff = series[-1]
            
        X, y = self.supervise(series)
        
        Xtr_Xvl_ytr_yvl = self.train_val_split(X, y)
        
        if self.sclr is not None:
            Xtr_Xvl_ytr_yvl = self.scale(*Xtr_Xvl_ytr_yvl)
        
        return Xtr_Xvl_ytr_yvl
    
    
    def log_transform(self, series):
        """
        # Param
        - series (ndarray: (T, )): raw series
        
        # Return
        - logged (ndarray: (T, )): logged series
        """
        # only when 1d feature (reshape)
        logged = self.log_trnsfmr.transform(series.reshape(-1,1))
        # only when 1d feature
        logged = logged.flatten()
        
        return logged
            
    
    def difference(self, series):
        """
        # Param
        - seires (ndarray: (T, )): raw series
        
        # Return
        -  (ndarray: (T-1, )): diffed series
        """
        return series[1:] - series[:-1]
    
    def supervise(self, series):
        """
        # Param
        - series (ndarray: (T, ))
        
        # Return
        - X (ndarray: (T-1, )): input
        - y (ndarray: (T-1, )): label
        """
        X = series[:-1]
        y = series[1:]
        return X, y
    
    def train_val_split(self, X, y):
        """
        # Param
        - X (ndarray: (T, )): input
        - y (ndarray: (T, )): label
        
        # Return
        - X_train, X_val, y_train, y_val 
            (ndarray: (T_train/val, 1))
        """
        val_size = 0.3
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                         test_size=val_size, shuffle=False)
        # only when F = 1
        X_train = X_train.reshape(-1,1)
        X_val   =   X_val.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        y_val   =   y_val.reshape(-1,1)
        
        return X_train, X_val, y_train, y_val
    
    def scale(self, X_train, X_val, y_train, y_val):
        """
        # Param
        - X_train, X_val, y_train, y_val 
            (ndarray: (T, 1))
        
        # Return
        - X_train, X_val, y_train, y_val 
            (ndarray: (T, 1)): scaled
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
    


# In[ ]:

# TODO: test all combinations of diff x scale
def test_prcsr(log_trnsfmr, diff, sclr, ysclr):
    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    series = series[:102]
    print('raw', series.shape)
    print(series[:5])
    print()
    plt.plot(series)
    plt.show()
    
    prcsr = Processer(log_trnsfmr=log_trnsfmr, diff=diff, 
                      sclr=sclr, ysclr=ysclr)
    
    # log
    if prcsr.log_trnsfmr is not None:
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
    ds_train, ds_test = prcsr.get_datasets(series)


# In[ ]:

def loop_prc(func):
    search_space={
        'log':[True, False],
        'diff':[True, False],
        'sclr':['MinMax', 'MinMax+', 'Standard', None],
        'ysclr':[True, False]
    }


# In[ ]:

if __name__=="__main__":
    # configs
    configs = {
        'log_trnsfmr':FunctionTransformer(np.log1p),
        'diff':True,
        'sclr':MinMaxScaler(feature_range=(-1,1)),
        'ysclr':MinMaxScaler(feature_range=(-1,1)),
    } 
    test_prcsr(**configs)

