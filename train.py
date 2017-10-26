#!/usr/bin/env python


# coding: utf-8

# In[ ]:

import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable
from chainer.iterators import SerialIterator

# no need
from chainer import Chain


# In[ ]:

class DemoModel(Chain):
    def __init__(self):
        super(DemoModel, self).__init__(
            fc=L.Linear(1,1)
        )
        pass
    
    def __call__(self, x):
        return x


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

X = np.arange(7)[:, np.newaxis, np.newaxis].transpose((1,0,2)).astype(np.float32)
y = np.arange(7)[:, np.newaxis, np.newaxis].transpose((1,0,2)).astype(np.float32)

model = DemoModel()
model(X)


# In[ ]:



