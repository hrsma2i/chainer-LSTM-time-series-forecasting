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
        
        X_trvl = np.concatenate((X_train, X_val), axis=0)
        obs_trvl = np.concatenate((y_train, y_val), axis=0)
        
        self.n_val = y_val.shape[0]

        # predict
        pred_trvl = []
        model.reset_state()
        for Xt in X_trvl:
            p_t = model(Xt.reshape(-1, 1)).data[0]
            pred_trvl.append(p_t)
        pred_trvl = np.array(pred_trvl)
        
        self.obss['direct'] = obs_trvl.copy()
        self.preds['direct'] = pred_trvl.copy()
        

        # inverse transform
        if prcsr.ysclr is not None:
            obs_trvl = prcsr.inverse_scale(obs_trvl)
            pred_trvl = prcsr.inverse_scale(pred_trvl)
            
            self.obss['unscale'] = obs_trvl.copy()
            self.preds['unscale'] = pred_trvl.copy()


        if prcsr.diff:
            before_diff = series.copy()
            if prcsr.log:
                before_diff = prcsr.log_transform(before_diff)

            obs_trvl  = before_diff[2:] 

            obs1_trvl = before_diff[1:-1]
            # TODO just pred + obs1 and delete inverse_diff_given
            pred_trvl = prcsr.inverse_diff_given(pred_trvl, obs1_trvl)
            
            self.obss['undiff'] = obs_trvl.copy()
            self.preds['undiff'] = pred_trvl.copy()


        if prcsr.log:
            pred_trvl = prcsr.inverse_log(pred_trvl)
            obs_trvl = series[-pred_trvl.shape[0]:]
            
            self.obss['unlog'] = obs_trvl.copy()
            self.preds['unlog'] = pred_trvl.copy()
        
        self.obss['raw'] = obs_trvl.copy()
        self.preds['raw'] = pred_trvl.copy()
        
    def get_pred_train(self, key):
        return self.preds[key][:-self.n_val]
    
    def get_pred_val(self, key):
        return self.preds[key][-self.n_val:]
    
    def get_obs_train(self, key):
        return self.obss[key][:-self.n_val]
    
    def get_obs_val(self, key):
        return self.obss[key][-self.n_val:]


# In[ ]:

def plot_fitting(dict_arrays, title=None):
    """
    # Param
    - dict_arrays (dict):
        - key (str): used as the label when plotting
        - value (ndarray)
    """
    plt.figure()
    if title is not None:
        plt.title(title)
    for k, v in dict_arrays.items():
        plt.plot(v, label=k)
    plt.legend()


# In[ ]:

def compare_with_baseline(data_root, pred_baseline_root,
                          root, name_seq, name_prc):
    """
    - Score (quantitively compare)
    - Plot fitting (qualitatively compare)
    - data: test
    
    # Param
    
    """
    # observation
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    
    # pred baseline
    name_csv_baseline  = 'pred_baseline_{}.csv'.format(name_seq)
    path_csv_baseline  = os.path.join(pred_baseline_root, name_csv_baseline)
    pred_test_baseline = pd.read_csv(path_csv_baseline, 
                           header=None).values.flatten()
    
    # pred LSTM
    model, prcsr, path_csv_train = setup(data_root=data_root, 
                                         root=root,
                                         name_seq=name_seq,
                                         name_prc=name_prc)
    
    pred_test = predict(num_pred=len(obs_test), 
                        model=model, prcsr=prcsr,
                        path_csv_train=path_csv_train)
    
    # score
    scrr = Scorer(obs_test, pred_test, do_adjust=True)
    scrr_baseline = Scorer(obs_test, pred_test_baseline, do_adjust=True)
    scores = {
        'LSTM':scrr.get_all(),
        'baseline':scrr_baseline.get_all()
    }
    df_score = pd.DataFrame(scores)
    display(df_score)
    
    # plot fitting
    d_plot = {
        'obs':obs_test,
        'pred LSTM':pred_test,
        'pred baseline':pred_test_baseline,
    }
    plot_fitting(d_plot, title=name_seq)


# In[ ]:

# comparison with baseline
if __name__=="__main__":
    data_root = 'data'
    pred_baseline_root = 'data/pred_baseline'
    root      = 'result/test'
    name_seq  = 'airline'
    name_prc  = 'default'
    
compare_with_baseline(data_root, pred_baseline_root,
                          root, name_seq, name_prc)


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

if __name__=="__main__":
    data_root = 'data'
    root      = 'result/test'
    name_seq  = 'airline'
    name_prc  = 'not_log'
    
    model, prcsr, path_csv_train = setup(data_root=data_root, 
                                         root=root,
                                         name_seq=name_seq,
                                         name_prc=name_prc)
    # pred/obs train/val
    prdctr = Predictor(model=model, prcser=prcsr,
                       path_csv_train=path_csv_train)
    
    # obs test
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    # pred test
    pred_test = predict(num_pred=len(obs_test), 
                        model=model, prcsr=prcsr,
                        path_csv_train=path_csv_train)
    
    for k in prdctr.preds.keys():
        d_plot = {
            'obs':prdctr.get_obs_train(k),
            'pred':prdctr.get_pred_train(k),
        }
        plot_fitting(d_plot, title='train_'+k)
        
        d_plot = {
            'obs':prdctr.get_obs_val(k),
            'pred':prdctr.get_pred_val(k),
        }
        plot_fitting(d_plot, title='val_'+k)
        
    d_plot = {
        'obs':pred_test,
        'pred':obs_test,
    }
    plot_fitting(d_plot, title='test_'+k)


# In[ ]:

# comparison with baseline
if __name__=="__main__":
    pred_baseline_root = 'data/pred_baseline'
    data_root = 'data'
    root      = 'result/test'
    name_seq  = 'airline'
    name_prc  = 'not_log'
    
    model, prcsr, path_csv_train = setup(data_root=data_root, 
                                         root=root,
                                         name_seq=name_seq,
                                         name_prc=name_prc)
    # pred/obs train/val
    prdctr = Predictor(model=model, prcser=prcsr,
                       path_csv_train=path_csv_train)
    
    # obs test
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    # pred test
    pred_test = predict(num_pred=len(obs_test), 
                        model=model, prcsr=prcsr,
                        path_csv_train=path_csv_train)
    
    for k in prdctr.preds.keys():
        d_plot = {
            'obs':prdctr.get_obs_train(k),
            'pred':prdctr.get_pred_train(k),
        }
        plot_fitting(d_plot, title='train_'+k)
        
        d_plot = {
            'obs':prdctr.get_obs_val(k),
            'pred':prdctr.get_pred_val(k),
        }
        plot_fitting(d_plot, title='val_'+k)
        
    d_plot = {
        'obs':pred_test,
        'pred':obs_test,
    }
    plot_fitting(d_plot, title='test_'+k)


# In[ ]:



