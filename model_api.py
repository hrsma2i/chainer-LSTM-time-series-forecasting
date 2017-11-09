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
    print('no module "jupyterthemes"')


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

def last_epoch(root):
    """
    Select epoch in which the model have best val loss
    
    # Param
    - root (str): path where the model's weights each epoch are
    
    # Return
    - epoch (int): best epoch
    """
    path_log = os.path.join(root, 'log')
    df_log = pd.read_json(path_log)
    
    best_idx = df_log['epoch'].argmax()
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
    
    series = pd.read_csv(path_csv_train, header=None).values.astype(np.float)
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

def setup_full(data_root, root, name_seq, name_prc, epoch=None):
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
    name_hp = 'full_'+select_hp(root=path_prc)
    path_hp = os.path.join(path_prc, name_hp)
    if epoch is None:
        epoch = last_epoch(root=path_hp)
    print(path_hp)
    print('epoch', epoch)
    model = get_learned_model(root=path_hp, epoch=epoch)
    
    return model, prcsr, path_csv_train


# In[ ]:

class Predictor(object):
    def __init__(self, model, prcsr, path_csv_train):
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
        if v is not None:
            plt.plot(v, label=k)
    plt.legend()
    plt.show()


# In[ ]:

def compare_full(data_root, pred_baseline_root,
                          root, name_seq, name_prc, 
                          lstm_epoch=None):
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
    model, prcsr, path_csv_train = setup_full(data_root=data_root, 
                                         root=root,
                                         name_seq=name_seq,
                                         name_prc=name_prc,
                                         epoch=lstm_epoch
                                        )
    
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

def compare_with_baseline(data_root, pred_baseline_root,
                          root, name_seq, name_prc, 
                          lstm_epoch=None):
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
                                         name_prc=name_prc,
                                        )
    
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

# test compare_full
if __name__=="__main__":
    for epoch in range(1,150+1, 15):
        data_root = 'data'
        root      = 'result/test'
        name_seq  = 'winnebago'
        name_prc  = 'default'
        pred_baseline_root = 'data/pred_baseline'
        compare_full(data_root=data_root,
                              pred_baseline_root=pred_baseline_root,
                              root=root,
                              name_seq=name_seq,
                              name_prc=name_prc,
                              lstm_epoch=epoch
                             )


# In[ ]:

# test compare_full
if __name__=="__main__":
    data_root = 'data'
    root      = 'result/test'
    name_seq  = 'winnebago'
    name_prc  = 'default'
    pred_baseline_root = 'data/pred_baseline'
    compare_with_baseline(data_root=data_root,
                          pred_baseline_root=pred_baseline_root,
                          root=root,
                          name_seq=name_seq,
                          name_prc=name_prc,
                         )


# In[ ]:

def same_len(*arrs):
    lengths = [arr.shape[0] for arr in arrs]
    lmin = min(lengths)
    cuts = tuple(arr[-lmin:] for arr in arrs)
    return cuts


# In[ ]:

def verify_prc(data_root, root, name_seq, name_prc, verbose=False):
    # obs test (must be written earier than setup_and_predict)
    name_csv_test  = '{}_test.csv'.format(name_seq)
    path_csv_test  = os.path.join(data_root, name_csv_test)
    obs_test = pd.read_csv(path_csv_test, 
                           header=None).values.flatten()
    
    def setup_and_predict(name_prc):
        # setup
        model, prcsr, path_csv_train = setup(data_root=data_root, 
                                             root=root,
                                             name_seq=name_seq,
                                             name_prc=name_prc)
        # pred/obs train/val
        prdctr = Predictor(model=model, prcsr=prcsr,
                           path_csv_train=path_csv_train)
        # pred test
        pred_test = predict(num_pred=len(obs_test), 
                            model=model, prcsr=prcsr,
                            path_csv_train=path_csv_train)
        return prdctr, pred_test
    
    
    prdctr, pred_test = setup_and_predict(name_prc)
    prdctr_def, pred_test_def = setup_and_predict('default')
    
    
    def score_and_plot(trvlts, obs, pred, pred_def, key='raw'):
        # score
        scrr = Scorer(obs, pred, do_adjust=True)
        scrr_def = Scorer(obs, pred_def, do_adjust=True)
        scores = {
            name_prc:scrr.get_all(),
            'defult':scrr_def.get_all()
        }
        df_score = pd.DataFrame(scores)
        display(df_score)
        
        # plot fitting
        d_plot = {
            'obs':obs,
            name_prc:pred,
            'dafault':pred_def, 
        }
        plot_fitting(d_plot, title=trvlts+' '+key)
        
    def plot_inverse(name, prdctr):
        keys = prdctr.preds.keys()
        for k in keys:
            print(name, k)
            # train
            obs_train = prdctr.get_obs_train(k)
            pred_train = prdctr.get_pred_train(k)
            d_plot = {
                'obs':obs_train,
                name:pred_train,
            }
            plot_fitting(d_plot, title=name+' train '+k)
            # val
            obs_val = prdctr.get_obs_val(k)
            pred_val = prdctr.get_pred_val(k)
            d_plot = {
                'obs':obs_val,
                name:pred_val,
            }
            plot_fitting(d_plot, title=name+' val '+k)
    
    k = 'raw'
    # train
    pred_train = prdctr.get_pred_train(k)
    obs_train = prdctr_def.get_obs_train(k)
    pred_train_def = prdctr_def.get_pred_train(k)
    pred_train, obs_train, pred_train_def = same_len(pred_train,
                                                     obs_train,
                                                     pred_train_def)
    score_and_plot('train', obs=obs_train,
                   pred=pred_train,
                   pred_def=pred_train_def, key=k)

    # val
    pred_val = prdctr.get_pred_val(k)
    obs_val = prdctr_def.get_obs_val(k)
    pred_val_def = prdctr_def.get_pred_val(k)
    pred_val, obs_val, pred_val_def = same_len(pred_val,
                                                     obs_val,
                                                     pred_val_def)
    score_and_plot('val', obs=obs_val,
                   pred=pred_val,
                   pred_def=pred_val_def, key=k)
        
    # test
    pred_test, obs_test, pred_test_def = same_len(pred_test,
                                                     obs_test,
                                                     pred_test_def)
    score_and_plot('test', obs=obs_test,
                   pred=pred_test,
                   pred_def=pred_test_def, key=k)
    
    if verbose:
        plot_inverse('default', prdctr_def)
        plot_inverse(name_prc,  prdctr)


# In[ ]:

# test verify_prc
if __name__=="__main__":
    data_root = 'data'
    root      = 'result/test'
    name_prc  = 'not_diff'
    name_seq  = 'toy'
    
    verify_prc(data_root=data_root, root=root,
               name_seq=name_seq, name_prc=name_prc,
               verbose=True)


# In[ ]:

def main_compare():
    pred_baseline_root = 'data/pred_baseline'
    data_root = 'data'
    name_sequences = 'sequences'
    root      = 'result/test'
    name_prc  = 'default'
    
    path_sequences = os.path.join(data_root, name_sequences)
    seqs = [ seq.rstrip()
        for seq in open(path_sequences, 'r').readlines()]
    
    for name_seq in seqs:
        compare_with_baseline(data_root=data_root,
                              pred_baseline_root=pred_baseline_root,
                              root=root, name_seq=name_seq,
                              name_prc=name_prc)


# In[ ]:

def main_verification(name_prc, verbose):
    data_root = 'data'
    root      = 'result/test'
    name_sequences = 'sequences_verify'
    
    path_sequences = os.path.join(data_root, name_sequences)
    seqs = [ seq.rstrip()
        for seq in open(path_sequences, 'r').readlines()]
    
    for name_seq in seqs:
        verify_prc(data_root=data_root, root=root,
                   name_seq=name_seq, name_prc=name_prc,
                   verbose=verbose)


# In[ ]:

# test main_verification
if __name__=="__main__":
    prcs = [
        'not_log',
        'not_diff',
        'minmax+',
        'standard',
        'not_scale',
        'not_label_scale',
    ]
    for name_prc in prcs:
        main_verification(name_prc=name_prc, verbose=True)


# In[ ]:

if __name__=="__main__":
    main_compare()


# In[ ]:



