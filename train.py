#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import chainer.links as L
import chainer.functions as F
import chainer
from chainer import Variable, training, optimizers, reporter
from chainer.training import extensions
from chainer.iterators import SerialIterator

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

# def train():
    
def train(units, opt, n_epoch, out):
    dict_opt = {
        'SGD':optimizers.SGD(),
        'Adam':optimizers.Adam(),
        'RMSprop':optimizers.RMSprop(),
        'AdaDelta':optimizers.AdaDelta(),
        'NesterovAG':optimizers.NesterovAG(),
        'MomentumSGD':optimizers.MomentumSGD(),
    }
    
    optimizer = dict_opt[opt]
    
    model = LossSumMSEOverTime(RNN(units))
    
    optimizer.setup(model)
    
    series = pd.read_csv('data/airline_train.csv', header=None).values.flatten()
    if series.ndim == 1:
        print('features = 1')
        series = series.reshape(-1, 1)
    prcsr = Processer()
    ds_train, ds_val = prcsr.get_datasets(series)
    
    itr_train = SerialIterator(ds_train, batch_size=1, shuffle=False)
    itr_val   = SerialIterator(ds_val  , batch_size=1, shuffle=False, repeat=False)
    
    updater = UpdaterRNN(itr_train, optimizer)
    
    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=out)
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
    trainer.extend(extensions.ProgressBar())
    
    trainer.run()


# In[ ]:

if __name__=="__main__":
    out = 'result'
    n_epoch = 100
    units = (2,3)
    opt = 'Adam'


# In[ ]:



