#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import os
import json
from itertools import product, chain, islice
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
import chainer.links as L
import chainer.functions as F
import chainer
from chainer import Variable, training, optimizers, reporter
from chainer.training import extensions, util
from chainer.iterators import SerialIterator
from sklearn.model_selection import ParameterGrid, ParameterSampler

from model import RNN
from data_process import Processer


# In[ ]:

class LossSumMSEOverTime(L.Classifier):
    def __init__(self, predictor):
        super(LossSumMSEOverTime, self).__init__(predictor, lossfun=F.mean_squared_error)
    
    def __call__(self, X_STF, y_STF):
        """
        # Param
        - X_STF (Variable: (S, T, F))
        - y_STF (Variable: (S, T, F))
        S: samples
        T: time_steps
        F: features
        
        # Return
        - loss (Variable: (1, ))
        """
        X_TSF = X_STF.transpose(1,0,2)
        y_TSF = y_STF.transpose(1,0,2)
        seq_len  = X_TSF.shape[0]
        
        loss = 0
        
        for t in range(seq_len):
            pred = self.predictor(X_TSF[t])
            obs  = y_TSF[t]
            loss += self.lossfun(pred, obs)
        loss /= seq_len
        
        reporter.report({'loss': loss}, self)
        
        return loss


# In[ ]:

class UpdaterRNN(training.StandardUpdater):
    def __init__(self, itr_train, optimizer, device=-1):
        super(UpdaterRNN, self).__init__(itr_train, optimizer, device=device)
        
    # overrided
    def update_core(self):
        itr_train = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        
        batch = itr_train.__next__()
        X_STF, y_STF = chainer.dataset.concat_examples(batch, self.device)
        
        optimizer.target.zerograds()
        optimizer.target.predictor.reset_state()
        loss = optimizer.target(Variable(X_STF), Variable(y_STF))
        
        loss.backward()
        optimizer.update()


# In[ ]:

# TODO 連続で勾配がeps以上上昇したら
class ExploasionStoppingTrigger(object):

    def __init__(self, max_epoch, key, stop_condition=None, 
                 eps=10, trigger=(1, 'epoch')):
        self.max_epoch = max_epoch
        self.eps = eps
        self._key = key
        self._current_value = None
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self.stop_condition = stop_condition or self._stop_condition

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.
        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
                this iteration.
        """

        epoch_detail = trainer.updater.epoch_detail
        if self.max_epoch <= epoch_detail:
            print('Reached to max_epoch.')
            return True

        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._current_value is None:
            self._current_value = value
            return False
        else:
            if self.stop_condition(self._current_value, value):
                # print('Previous value {}, Current value {}'
                #       .format(self._current_value, value))
                print('Invoke ExploasionStoppingTrigger...')
                self._current_value = value
                return True
            else:
                self._current_value = value
                return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def _stop_condition(self, current_value, new_value):
        return new_value - current_value > self.eps


# In[ ]:

def hp2name(hp):
    d = {
        'u':hp['units'],
        'opt':hp['optimizer'].__class__.__name__
    }
    name = '_'.join([k+str(v) for k, v in d.items()])
    return name


# In[ ]:

def hp2json(hp):
    hp_json = {
        'units':hp['units'],
        'optimizer':hp['optimizer'].__class__.__name__
    }
    return hp_json


# In[ ]:

def train(datasets, hp, out, n_epoch):
    """
    dump the given hyperparameters hp, and
    train a model with the hyperparameters
    
    # Param
    - datasets (tuple): ds_train, ds_val
    - hp (dict): hyperparameters
    - out (str): the path where "log" and "hyperparameters"
                will be dumped
    - n_epoch: up to which train the model
    """
    # dump hyperparameters
    if not os.path.exists(out):
        os.mkdir(out)
    path_json = os.path.join(out, 'hyperparameters.json')
    hp_json = hp2json(hp)
    json.dump(hp_json, open(path_json, 'w'))
    
    
    # training
    units = hp['units']
    optimizer = hp['optimizer']
    
    model = LossSumMSEOverTime(RNN(units))
    
    optimizer.setup(model)
    
    
    ds_train, ds_val = datasets
    itr_train = SerialIterator(ds_train, batch_size=1, shuffle=False)
    itr_val   = SerialIterator(ds_val  , batch_size=1, shuffle=False, repeat=False)
    
    updater = UpdaterRNN(itr_train, optimizer)
    
    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    expl_stop = ExploasionStoppingTrigger(n_epoch, 
                        key='validation/main/loss', eps=10)
    trainer = training.Trainer(updater, 
                               stop_trigger=expl_stop, out=out)
    trainer.extend(extensions.Evaluator(
                itr_val, eval_model, device=-1,
                eval_hook=lambda _: eval_rnn.reset_state()))
    
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot_object(model.predictor, 
                                               filename='model_epoch-{.updater.epoch}'))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                        x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PrintReport(
                    ['epoch','main/loss','validation/main/loss']
                ))
    
    trainer.run()


# In[ ]:

def tune(root, datasets, n_sample=10, n_epoch=5):
    # search space
    max_n_layer = 5
    max_n_unit  = 5
    opts = [
        optimizers.SGD(),
        optimizers.Adam(),
        optimizers.RMSprop(),
        optimizers.AdaDelta(),
        optimizers.NesterovAG(),
        optimizers.MomentumSGD(),
    ]
    
    result = []
    
    for i in range(n_sample):
        n_layer = np.random.randint(1, max_n_layer+1)
        units = tuple(
                    np.random.randint(1, max_n_unit+1)
                    for l in range(n_layer)
                )
        optimizer = np.random.choice(opts)
        
        hp = {
            'units': units,
            'optimizer': optimizer,
        }
        path_hp = os.path.join(root, hp2name(hp))
        
        hp_json = hp2json(hp)
        result.append(hp_json)
        
        display('sample{}'.format(i), pd.Series(hp_json))
        print('path_hp', path_hp)
        print()
        
        # training
        train(datasets=datasets, hp=hp, 
              out=path_hp, n_epoch=n_epoch)
        
        print(''.join(['-' * 60]))

    df = pd.DataFrame(result)
    display(df)


# In[ ]:

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def loop_prc(root, series):
    """
    # Param
    - root   (str): the path where results will be saved
    - series (ndarray: (T, F)):
    """
    def routin(name_prc, prscr, root=root, series=series):
        path_prc = os.path.join(root, name_prc)
        datasets = prcsr.get_datasets(series)
        tune(root=path_prc, datasets=datasets)
        
    name_prc = 'not_log'
    prcsr = Processer(log=False)
    routin(name_prc, prcsr)
        
    name_prc = 'not_diff'
    prcsr = Processer(diff=False)
    routin(name_prc, prcsr)
    
    name_prc = 'not_diff'
    prcsr = Processer(diff=False)
    routin(name_prc, prcsr)


# In[ ]:

def loop_seq(root, prcsr=Processer()):
    seqs = [
        'airline',
        'car',
        'companyx',
        'house',
        'winnebago'
    ]
    
    for name_seq in seqs:

        series = pd.read_csv('data/{}_train.csv'.format(name_seq)
                             , header=None).values.flatten()
        if series.ndim == 1:
            print('features = 1')
            series = series.reshape(-1, 1)

        path_seq = os.path.join(root, name_seq)
        print(path_seq)
        if not os.path.exists(path_seq):
            os.mkdir(path_seq)

        datasets = prcsr.get_datasets(series)

        tune(root=path_seq, datasets=datasets)


# In[ ]:

if __name__=="__main__":
    root = 'result/test_seq_loop'
    prcsr = Processer()
    loop_seq(root=root, prcsr=prcsr)
    # 


# In[ ]:

# train a model
if __name__=="__main__":
    #prcsr = Processer(log_trnsfmr=log_trnsfmr, diff=diff, 
    #                  sclr=sclr, ysclr=ysclr)
    lr = 0.01
    prcsr = Processer()

    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    if series.ndim == 1:
        print('ndim = 1')
        series = series.reshape(-1, 1)

    hp = {
        'units':(3, 7, 3),
        'optimizer':optimizers.Adam(alpha=lr)
    }    
    print(hp['optimizer'].alpha)
    

    root = 'result/test/adam{}'.format(lr)

    datasets = prcsr.get_datasets(series)
    
    # training
    #train(datasets, hp, out=root, n_epoch=300)

