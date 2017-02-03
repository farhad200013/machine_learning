from skimage.feature import local_binary_pattern
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

accuracy=[]

filelist=glob.glob('GTSRB_subset/*/*.jpg')
#skimage.feature.local_binary_pattern
## settings for LBP
radius = 3
n_points = 8 * radius
X=[]
Y=[]
#
#
for i in filelist:
    image=plt.imread(i)
   
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)




clflist = [KNeighborsClassifier(n_neighbors=6), LDA(), SVC()]


for l in clflist:
    l.fit(X_train,y_train)
    y_pred=l.predict(X_test)
    accuracy_score(y_test, y_pred)


