
# coding: utf-8

# In[ ]:


import time

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from IPython.display import display
import codecs


# In[ ]:


if __name__=="__main__":
    df = pd.read_csv('../data/airline.csv', delimiter=',')
    series = df['Passengers'].values
    train = np.savetxt('../data/airline_train.csv', series[:-12], fmt="%d")
    test = np.savetxt('../data/airline_test.csv', series[-12:], fmt="%d")
    

