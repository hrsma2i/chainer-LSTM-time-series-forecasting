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
try:
    from jupyterthemes import jtplot
    jtplot.style(
        theme='grade3',
        figsize=(20, 10),
        fscale=2,
    )
except ModuleNotFoundError:
    pass


from data_process import Processer, name2prc
from model import RNN
# change below later
from others.score import Scorer


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
    
    series = pd.read_csv(path_csv_train, header=None).values
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

# comparison with baseline
if __name__=="__main__":
    data_root = 'data'
    pred_baseline_root = 'data/pred_baseline'
    root      = 'result/test'
    name_seq  = 'car'
    name_prc  = 'default'
    
    # observation
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    
    # pred baseline
    name_csv_test  = 'pred_baseline_{}.csv'.format(name_seq)
    path_csv_test  = os.path.join(pred_baseline_root, name_csv_test)
    pred_test_baseline = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    
    # pred LSTM
    name_csv_train = '{}_train.csv'.format(name_seq)
    path_csv_train = os.path.join(data_root, name_csv_train)
    
    prcsr = Processer(**name2prc(name_prc))
    
    path_seq = os.path.join(root, name_seq)
    path_prc = os.path.join(path_seq, name_prc)
    
    name_hp = select_hp(root=path_prc)
    
    path_hp = os.path.join(path_prc, name_hp)
    epoch = select_epoch(root=path_hp)
    print(path_hp)
    print('epoch', epoch)
    
    model = get_learned_model(root=path_hp, epoch=epoch)
    
    pred_test = predict(num_pred=12, model=model, prcsr=prcsr,
                        path_csv_train=path_csv_train)
    
    scrr = Scorer(obs_test, pred_test, do_adjust=True)
    scrr_baseline = Scorer(obs_test, pred_test_baseline, do_adjust=True)
    scores = {
        'LSTM':scrr.get_all(),
        'baseline':scrr_baseline.get_all()
    }
    df_score = pd.DataFrame(scores)
    display(df_score)
    
    # plot fitting
    plt.figure(figsize=(20,10))
    plt.plot(obs_test,  label='obs')
    plt.plot(pred_test, label='pred LSTM')
    plt.plot(pred_test_baseline, label='pred baseline')
    plt.legend()


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
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    
    print(name_hp)
    print(epoch)
    plt.figure(figsize=(20,10))
    plt.plot(pred_test, label='pred')
    plt.plot(obs_test,  label='obs')
    plt.legend()


# In[ ]:

def setup(data_root, root, name_seq, name_prc):
    """
    Make variables to initialize Predictor
    """
    
    # setup path_csv_train
    name_csv_train = '{}_train.csv'.format(name_seq)
    path_csv_train = os.path.join(data_root, name_csv_train)

    # setup processer 
    prcsr = Processer(**name2prc(name_prc))

    # setup model
    path_seq = os.path.join(root, name_seq)
    path_prc = os.path.join(path_seq, name_prc)
    name_hp = select_hp(root=path_prc)
    path_hp = os.path.join(path_prc, name_hp)
    epoch = select_epoch(root=path_hp)
    print(path_hp)
    print('epoch', epoch)
    model = get_learned_model(root=path_hp, epoch=epoch)
    
    return model, prcsr, path_csv_train


# In[ ]:

class Predictor(object):
    def __init__(self, model, prcser, path_csv_train):
        self.obss  = {}
        self.preds = {}
        
        series = pd.read_csv(path_csv_train, header=None).values
        X_train, X_val, y_train, y_val = prcsr.transform_train(series)
        
        X_train = np.concatenate((X_train, X_val), axis=0)
        obs_train = np.concatenate((y_train, y_val), axis=0)
        
        self.n_train = y_train.shape[0]

        # predict
        pred_train = []
        model.reset_state()
        for Xt in X_train:
            p_t = model(Xt.reshape(-1, 1)).data[0]
            pred_train.append(p_t)
        pred_train = np.array(pred_train)
        
        self.obss['direct'] = obs_train.copy()
        self.preds['direct'] = pred_train.copy()
        

        # inverse transform
        if prcsr.ysclr is not None:
            obs_train = prcsr.inverse_scale(obs_train)
            pred_train = prcsr.inverse_scale(pred_train)
            
            self.obss['unscale'] = obs_train.copy()
            self.preds['unscale'] = pred_train.copy()


        if prcsr.diff:
            before_diff = series.copy()
            if prcsr.log:
                before_diff = prcsr.log_transform(before_diff)

            obs_train  = before_diff[2:] 

            obs1_train = before_diff[1:-1]
            # TODO just pred + obs1 and delete inverse_diff_given
            pred_train = prcsr.inverse_diff_given(pred_train, obs1_train)
            
            self.obss['undiff'] = obs_train.copy()
            self.preds['undiff'] = pred_train.copy()


        if prcsr.log:
            pred_train = prcsr.inverse_log(pred_train)
            obs_train = series[-pred_train.shape[0]:]
            
            self.obss['unlog'] = obs_train.copy()
            self.preds['unlog'] = pred_train.copy()
        
        self.obss['raw'] = obs_train.copy()
        self.preds['raw'] = pred_train.copy()
        
def plot(self):
    preds = self.preds
    obss = self.obss
    for key in preds.keys():
        self.plot_each(key, obss[key], preds[key])

def plot_each(self, title, obs, pred):
    plt.figure()
    plt.title(title)
    plt.plot( obs, label='obs' )
    plt.plot(pred, label='pred')
    plt.axvline(self.n_train, color='red')
    plt.legend()

def plot_raw(self):
    preds = self.preds
    obss = self.obss
    key = 'raw'
    self.plot_each(key, obss[key], preds[key])


# In[ ]:

if __name__=="__main__":
    data_root = 'data'
    root      = 'result/test'
    name_seq  = 'car'
    name_prc  = 'not_scale'
    
    config = setup(data_root=data_root, root=root,
                   name_seq=name_seq, name_prc=name_prc)
    display(config)


# In[ ]:




# In[ ]:



