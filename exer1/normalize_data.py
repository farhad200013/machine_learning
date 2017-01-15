# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:27:52 2017

@author: aliTakin
"""

# this is task 5
import numpy as np


def normalize_data(X):
    
    [x1,x2]=np.shape(X)
    
    X_norm=[] 
    for i in  x2 :
       X_norm[:,i]= X[:,i]-np.mean(X[:,i], axis = 0)
       X_norm[:,i]= X[:,i]/np.std(X[:,i], axis = 0)
        
    return X_norm
      


