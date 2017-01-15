# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:25:14 2017

@author: aliTakin
"""

# this is task 4 

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
 
mat = loadmat(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\twoClassData.mat")

print(mat.keys()) # Which variables mat contains?
X = mat["X"] # Collect the two variables.
y = mat["y"].ravel()

X_zero = X[y==0, :]
X_one = X[y==1, :]
plt.plot(X_zero[:, 0], X_zero[:, 1], 'ro')
plt.plot(X_one[:, 0], X_one[:, 1], 'bo')
plt.show()

    
           
def normalize_data(X):
        
        return(X - X.mean(axis = 0)) / X.std(axis = 0)   



X_norm = normalize_data(X)

X_norm = normalize_data(X)
print(np.mean(X_norm, axis = 0)) # Should be 0
print(np.std(X_norm, axis = 0)) # Should be 1