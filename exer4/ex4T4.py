# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 13:04:54 2017
@author: aliTakin
"""

#read the data GTSRB 
from skimage.feature import local_binary_pattern
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


radius = 5
n_points = 8
X=[]
y=[]

filelist=glob.glob('GTSRB_subset/class1/*.jpg')
filelist2=glob.glob('GTSRB_subset/class2/*.jpg')

class_folders=sorted(glob.glob('GTSRB_subset/*'))
for i , folder in enumerate(class_folders):
    names=glob.glob(folder+'/*')
    for name in names:
        image=image=plt.imread(name)
        localbp = local_binary_pattern(image, n_points,radius)
        histogram = np.histogram(localbp, bins=range(257))[0]
        X.append(histogram)
        y.append(i)
    

#skimage.feature.local_binary_pattern
## settings for LBP



#
#

 

#print(X)
 



   
### question 5, 
#accuracy=[]   
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)




clflist = [KNeighborsClassifier(n_neighbors=6), LDA(), SVC()]





clflist[2].fit(X_train,y_train)
y_pred=clflist[2].predict(X_test)
accuracy=accuracy_score(y_test,y_pred)



#
#
#for l in clflist:
#    l.fit(X_train,y_train)
#    y_pred=l.predict(X_test)
#    accuracy= accuracy_score(y_test, y_pred)


    
#def overlay_labels(filelist, lbp, labels):
#    mask = np.logical_or.reduce([lbp == each for each in labels])
#    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)
#
#
#def highlight_bars(bars, indexes):
#    for i in indexes:
#        bars[i].set_facecolor('r')
#
#
##image = data.load(filelist)
#lbp = local_binary_pattern(image, n_points, radius, METHOD)
#
#
#def hist(ax, lbp):
#    n_bins = lbp.max() + 1
#    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
#                   facecolowr='0.5')