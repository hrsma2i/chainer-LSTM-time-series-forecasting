
# coding: utf-8

# In[ ]:


import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable, datasets, iterators, optimizers
from chainer import report, training
from chainer.training import extensions

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# HP
# architecture: n_layer
# in each layer: n_unit, activation? (all of them depends on n_layer)
# training: n_epoch, optimization, lr

class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__()
        n_in = 1
        n_unit = 10
        n_out = 1
        lstms = [('lstm', L.LSTM(n_in, n_unit))]
        self.lstms = lstms
        for name, lstm in lstms:
            self.add_link(name, lstm)
        self.add_link('fc', L.Linear(n_unit, n_out))
        
    def __call__(self, x):
        h = Variable(x)
        for name,lstm in self.lstms:
            h = lstm(h)
        return self.fc(h)
    
    def reset_state():
        for name, lstm in self.lstms:
            lstm.reset_state()


# In[ ]:


if __name__=="__main__":
    model = RNN()
    X = np.array([[2]]).astype(np.float32)
    #X = Variable(X)
    model(X).data


# In[ ]:




