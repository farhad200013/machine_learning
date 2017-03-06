# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 07:40:29 2017

@author: aliTakin
"""
import numpy as np

import matplotlib.pyplot as plt
from scipy.io import loadmat as sio
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV


data=sio('arcene.mat')
y_train=np.ravel(data['y_train'])
y_test=np.ravel(data['y_test'])
X_train=data['X_train']
X_test=data['X_test']


parameters =     {
        'penalty': ['l1'],
        'C': np.logspace(-3, 5)
    } 
clf = GridSearchCV(estimator=LogisticRegression(), 
                                        param_grid = parameters,
                                        cv = 10  )





clf.fit(X_train, y_train.flatten())
params = clf.best_params_
print("\nBest parameters are:", params)
    
  
lr = LogisticRegression(penalty = params['penalty'], C = params['C'])
lr.fit(X_train, y_train)
print(" selected features:", np.count_nonzero(lr.coef_))
    
   
score_lr = accuracy_score(y_test.flatten(), lr.predict(X_test))
print("LR accuracy score :", score_lr)
    
