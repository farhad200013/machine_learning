"""
Created on Thu Jun  9 10:23:13 2016

@author: ali
"""

import numpy as np 
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import os
from sklearn.cross_validation import train_test_split
from scipy.misc import imrotate
import random 
from PIL import Image
import glob         

def normalize_image(img):
    
    img = img.astype(float)
    img -= img.min()
    
    if img.max() > 0:
        img /= img.max()
    else: # The image is all zeros: let's mark this by making the result gray.
        img = 0.5 * np.ones_like(img)
    
    return img

def parse_tif(filePath, numFramesPerTif = 70):
    
    img = Image.open(filePath)
    
    target = []
    
    for i in range (numFramesPerTif):
        try:
            img.seek(i)
            #img.tell
            target.append(np.array(img))
        except EOFError: #end of file error
            pass
    return np.array(target)   

X=[]
y=[]

labels = []
i = 0
      

for dataset in ["01", "02"]:
    datapath = '/home/ali/Downloads/N2DL-HeLa (2)/N2DL-HeLa/%s' % dataset
    print "Finding files from %s" % datapath 
    files=glob.glob(datapath + "/*.tif") # t01.tif
    ann_path = datapath + "_GT/SEG/" # man_seg01.tif
    
    print "Found %d files." % len(files)
    for name in files:
    
        basename = os.path.basename(name)    
        targetname = ann_path + "man_seg" + basename[1:]
        
        if not os.path.isfile(targetname):
            print "Target file %s not found." % targetname
            continue 
        
        
#        image=parse_tif(name)
#        image=np.mean(image, axis = 0)
        image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (128,128))       
        image = image.astype(np.float32) #sequential model is working with float32
        image -= np.min(image)
        image /= np.max(image)
        image = image[np.newaxis, :, :]
        X.append(image)
        labels.append(i)
    #    print "%d images collected..." % len(X)
        
        #target=parse_tif(targetname)   
        #target=np.any(target, axis = 0).astype(np.uint8)
        target = cv2.imread(targetname,cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (16,16))
        target = target.astype(np.float32)
        target = np.clip(target, 0, 1)
        target = target[np.newaxis, ...]
        
        y.append(target)
    
        flipped_X = image[:, ::-1, :]
        X.append(flipped_X)
        labels.append(i)
        
        flipped_y = target[:, ::-1, :]
        y.append(flipped_y)
    
        for angle in [90, 180, 270]:
            rotated_X = imrotate(image[0, ...], angle, interp = 'nearest')
            rotated_X = rotated_X.astype(np.float32)
            rotated_X /= np.max(rotated_X)
            rotated_y = np.clip(imrotate(target[0, ...], angle, interp = 'nearest'), 0, 1)
            
            X.append(rotated_X[np.newaxis])
            labels.append(i)
            y.append(rotated_y[np.newaxis])
            
        for angle in [90, 180, 270]:
            rotated_X = imrotate(flipped_X[0,...], angle, interp = 'nearest')
            rotated_X = rotated_X.astype(np.float32)
            rotated_X /= np.max(rotated_X)
            rotated_y = np.clip(imrotate(flipped_y[0, ...], angle, interp = 'nearest'), 0, 1)
            
            X.append(rotated_X[np.newaxis])
            labels.append(i)
            y.append(rotated_y[np.newaxis])
            
        i += 1
    
X = np.array(X)
y = np.array(y)
w, h = 3,3
model = Sequential()
model.add(Convolution2D(32, w, h, border_mode='same',input_shape=(1,128,128)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
 
model.add(Convolution2D(32, w, h, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Convolution2D(1, w, h, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('sigmoid'))


#model.add(Convolution2D(32, w, h, border_mode='same',input_shape=(3,32,32))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

#model.add(Flatten())
#model.add(Dense(256))
#model.add(Activation('relu'))
#model.add(Dense(2))
#model.add(Activation('softmax'))
 
print "Starting compilation..."
model.compile(loss='mean_absolute_error', optimizer='sgd')
    
print "Starting training..."

image_indices = np.unique(labels)
random.shuffle(image_indices)
train_file_indices = image_indices[:int(0.9 * len(image_indices))]
test_file_indices = image_indices[int(0.9 * len(image_indices)):]

X_train = []
X_test = []
y_train = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
y_test = []

for k in range(X.shape[0]):
    if labels[k] in train_file_indices:
        X_train.append(X[k,...])
        y_train.append(y[k,...])
    else:
        X_test.append(X[k,...])
        y_test.append(y[k,...])

X_train = np.array(X_train)       
y_train = np.array(y_train)       
X_test = np.array(X_test)       
y_test = np.array(y_test)       

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

training_errors = []
test_errors = []

for epoch in range(100):
    model.fit(X_train, y_train, batch_size=32, nb_epoch=1)
    y_hat = model.predict(X_test)
    test_error = np.mean(np.abs(y_hat - y_test))
    y_hat = model.predict(X_train)
    train_error = np.mean(np.abs(y_hat - y_train))
    
    training_errors.append(train_error)
    test_errors.append(test_error)
    print "Training error: %.4f" % train_error
    print "Test error: %.4f" % test_error

# Let's prepare illustration of how well the network segments

y_hat = model.predict(X_test)
for k in range(X_test.shape[0]):
    image = X_test[k,0]
    target = y_test[k, 0]
    prediction = y_hat[k, 0]

    image = normalize_image(image)
    target = normalize_image(target)    
    prediction = normalize_image(prediction)
    
    image[-16:, -16:] = target
    image[-16:, -32:-16] = prediction

    image = 255.0 * image / np.max(image)    
    image = image.astype(np.uint8)
    
    cv2.imwrite("result_images/img_%d.png" % k, image)
    
plt.figure()

plt.plot(training_errors, 'r-o', label = "Training error")
plt.plot(test_errors, 'b-o', label = "Test error")
plt.legend()
plt.savefig("FinalResults.pdf")
plt.show()
    
print "Training finished..."
 
