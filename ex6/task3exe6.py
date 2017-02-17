import glob
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D


radius = 3
n_points = 8 * radius

X = []
y = []


cls_fldrs = sorted(glob.glob('GTSRB_subset_2/*')) 
for i , folder in enumerate(cls_fldrs):  # i = 0 or 1  folder = ['GTSRB_subset\\class1', 'GTSRB_subset\\class2']
    names = glob.glob(folder + '/*jpg') 
    for name in names: 
        img = plt.imread(name)
        img = img.astype(np.double)
# if it is thiano       img = np.transpose(img)
        X.append(img) 
        y.append(i)

X = np.array(X)
y = np.array(y)

X = (X-np.min(X))/np.max(X)



print("X shape is %s" % str(X.shape))    
print("y shape is %s" % str(y.shape))


y = np_utils.to_categorical(y, 2)

     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)        

N = 10 # Number of feature maps
w, h = 3, 3 # Conv. window size
model = Sequential()
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        activation = 'relu',
                        input_shape = (64,64,3)))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(nb_filter = N,
                        nb_col = w,
                        nb_row = h,
                        activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size = 32, nb_epoch= 20, validation_data=(X_test, y_test))