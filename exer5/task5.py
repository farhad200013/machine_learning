import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from skimage.feature import local_binary_pattern
#from skimage import io

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


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
        
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifiers = [(RandomForestClassifier(), "Random Forest"),(ExtraTreesClassifier(), "Extra-Trees"),(AdaBoostClassifier(), "AdaBoost"),(GradientBoostingClassifier(), "GB-Trees")]
for clf, name in classifiers:
    
    clf.n_estimators = 100
    accuracies = []
    for iteration in range(100):
        
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_hat)
        accuracies.append(accuracy)
        print ("Accuracy for C = %f " %accuracy )



