import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from skimage.feature import local_binary_pattern
#from skimage import io

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import glob

radius = 3
n_points = 8 * radius

X = []
y = []



cls_fldrs = sorted(glob.glob('GTSRB_subset/*')) 
for i , folder in enumerate(cls_fldrs):  # i = 0 or 1  folder = ['GTSRB_subset\\class1', 'GTSRB_subset\\class2']
    names = glob.glob(folder + '/*jpg') 
    for name in names: 
        img = plt.imread(name)
        lbp = local_binary_pattern(img, n_points, radius) 
        histogram = np.histogram(lbp, bins = range(2**8+1))[0]
        X.append(histogram) 
        y.append(i)
        
X = preprocessing.scale(X) 
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_list = [LogisticRegression(), SVC()]
clf_name = ['LR', 'SVC']
C_range = 10.0 ** np.arange(-5, 0)

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print ("Accuracy for C = %.2e and penalty = %s is %.3f" % (C, penalty, score))



