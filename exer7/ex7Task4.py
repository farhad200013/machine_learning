# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:20:58 2017

@author: aliTakin
"""
import numpy as np
#from sklearn.svm import svc
import matplotlib.pyplot as plt
from scipy.io import loadmat as sio
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score


data=sio('arcene.mat')
y_train=np.ravel(data['y_train'])
y_test=np.ravel(data['y_test'])
X_train=data['X_train']
X_test=data['X_test']



rfecv = RFECV(estimator=LogisticRegression(), step=50, cv=10,
              scoring='accuracy')
rfecv.fit(X_train,y_train)
print(rfecv.n_features_)
lr1=LogisticRegression()
lr1.fit(X_train, y_train)
score_lr1 = accuracy_score(y_test, lr1.predict(X_test))
print(score_lr1)
plt.plot(range(0,10001,50), rfecv.grid_scores_)
