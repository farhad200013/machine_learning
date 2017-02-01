# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:52:45 2017

@author: aliTakin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split( digits.data, digits.target, test_size=0.2, random_state=0)

clf = KNeighborsClassifier(n_neighbors=6)
plt.gray()
plt.imshow(digits.images[0])
plt.show()

clf.fit(X_train, y_train)
# Testing code:
y_pridict=clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pridict, normalize=True, sample_weight=None)
print(accuracy)





