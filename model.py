#!/usr/bin/env python


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
    def __init__(self, units):
        super(RNN, self).__init__()
        
        n_in  = 1 # features
        n_out = 1
        
        #lstms =  [('lstm0', L.LSTM(n_in, units[0]))]
        #for l in range(1, len(units)):
        #    print(units[l-1], units[l])
        #    lstms.append(('lstm{}'.format(l), L.LSTM(units[l-1], units[l])))
        #
        #lstms =  [('lstm0', L.LSTM(n_in, units[0]))]
        #lstms += [('lstm{}'.format(l+1), L.LSTM(units[l], units[l+1]))
        #        for l in range(len(units)-1)]
        #
        lstms = [('lstm{}'.format(l), L.LSTM(None, n_unit))
                for l, n_unit in enumerate(units)]
        self.lstms = lstms
        for name, lstm in lstms:
            self.add_link(name, lstm)
            
        self.add_link('fc', L.Linear(units[-1], n_out))
        
        
    def __call__(self, x):
        """
        # Param
        - x (Variable: (S, F))
        S: samples
        F: features
        
        # Return
        -   (Variable: (S, 1))
        """
        h = x
        for name, lstm in self.lstms:
            h = lstm(h)
        return self.fc(h)
    
    def reset_state(self):
        for name, lstm in self.lstms:
            lstm.reset_state()


# In[ ]:

if __name__=="__main__":
    units = (2, 3, 5, 9)
    model = RNN(units)
    display(dir(model.children))


# In[ ]:



