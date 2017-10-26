
# coding: utf-8

# In[ ]:


import chainer.links as L
import chainer.functions as F
from chainer.iterators import SerialIterator


# In[ ]:


class LossSumMSEOverTime(L.Classifier):
    def __init__(self):
        super(LossSumMSEOverTime, self).__init__(predictor,
                                                lossfun=F.mean_squared_error)
    
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
        
        

