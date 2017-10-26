#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable, training
from chainer.iterators import SerialIterator


# In[ ]:

class LossSumMSEOverTime(L.Classifier):
    def __init__(self, predictor):
        super(LossSumMSEOverTime, self).__init__(predictor, lossfun=F.mean_squared_error)
    
    def __call__(self, X_STF, y_STF):
        """
        # Param
        - X_STF (ndarray: (S, T, F))
        - y_STF (ndarray: (S, T, F))
        S: samples
        T: time_steps
        F: features
        
        # Return
        - loss (Variable)
        """
        X_TSF = X_STF.transpose(1,0,2)
        y_TSF = y_STF.transpose(1,0,2)
        seq_len  = X_TSF.shape[0]
        
        loss = 0
        
        for t in range(seq_len):
            pred = self.predictor(Variable(X_TSF[t]))
            obs  = Variable(y_TSF[t])
            loss += self.lossfun(pred, obs)
        
        return loss


# In[ ]:

class UpadaterRNN(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, deivce):
        super(UpdaterRNN, self).__init__(train_iter, optimizer, device=device)
        
    # override
    def updater_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        
        X_STF, y_STF = train_iter.__next__()
        
        optimizer.target.predictor.reset_state()
        loss = optimizer.target(X_STF, y_STF)
        
        optimizer.target.zerograds()
        loss.backward()
        optimizer.update()


# In[ ]:



