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

from pymymod.chkprint import chkprint


# In[ ]:

class Processer(object):
    """
    # Attr
    - diff (bool): difference or not
    - scl  (str) : the method to scale
    - lbl  (bool): also scale lable or not
    - X_scaler (Scaler): scaler for input
    - y_scaler (Scaler): scaler for pred or label
    - last_raw  (int/float): last data of raw series
                            to compute ...
    - last_diff (int/float): last data of diffed series
                            to compute
    """
    def __init__(self, diff=True, scl='mimmax', lbl=True):
        self.diff = diff
        self.scl  = scl
        self.lbl  = lbl
        
    def transform_train(series):
        """
        # Param
        - series (ndarray: (T, )): raw time series
        T: time steps
        
        # Return
        - X_train (ndarray: (S, T, F))
        - X_val   (ndarray: (S, T, F))
        - y_train (ndarray: (S, T, F))
        - y_val   (ndarray: (S, T, F))
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
        self.last_raw = series[-1]
        
        if self.diff:
            series = self.difference(series)
            self.last_diff = series[-1]
            
        X, y = supervise(series)
        
        Xtr_Xvl_ytr_yvl = train_val_split(X, y)
        
        if self.scl:
            Xtr_Xvl_ytr_yvl = self.scale(*Xtr_Xvl_ytr_yvl)
            
        for seq in Xtr_Xvl_ytr_yvl:
            seq = seq[:, np.newaxis, np.newaxis]
            seq = seq.transpose(1, 0, 2)
        
        return Xtr_Xvl_ytr_yvl
    
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
            (ndarray: (T_train/val, ))
        """
        val_size = 0.3
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                         test_size=val_size, shuffle=False)
        return X_train, X_val, y_train, y_val


# In[ ]:

if __name__=="__main__":
    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    series = series[:102]
    chkprint('raw', series.shape)
    chkprint(series[:5])
    chkprint()
    
    diff = True
    scl  = 'minmax'
    lbl  = True
    
    prcsr = Processer(diff, scl, lbl)
    
    # diff
    if prcsr.diff:
        series = prcsr.difference(series)
        chkprint('diff', series.shape)
        chkprint(series[:5])
        chkprint()
        
    # supervise
    X, y = prcsr.supervise(series)
    chkprint('X', X.shape)
    chkprint(X[:5])
    chkprint()
    chkprint('y', y.shape)
    chkprint(y[:5])
    chkprint()
    
    # train val split
    Xtr_Xvl_ytr_yvl = prcsr.train_val_split(X, y)
    names = ('')
    for seq in Xtr_Xvl_ytr_yvl:
        chkprint(dir(seq))
        chkprint('X', seq.shape)
        chkprint(seq[:5])
        chkprint()


# In[ ]:




# In[ ]:



