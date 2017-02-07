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

import h5py

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

if __name__ == "__main__":
         
    X=[]
    y=[]
    
    labels = []
    i = 0
    target_dim = 128
    cache_file=""
    cache_file = "cache/cache_%d.h5" % target_dim
    
    if os.path.isfile(cache_file):
        with h5py.File(cache_file, "r") as fp:
            X = np.array(fp["X"])
            y = np.array(fp["y"])
            labels = list(fp["labels"])
            
    else:
        for dataset in ["01", "02" ,"03","04","05","06"]:
            #datapath = '/home/agedemo/src/Cell Semantic Segmentation/Data Set and code/N3DH-SIM/%s' % dataset
            datapath = '/home/ali/Downloads/N2DH-GOWT1 (2)/N2DH-GOWT1/%s' % dataset
           # print "Finding files from %s" % datapath 
            files=glob.glob(datapath + "/*.tif") # t01.tif
            ann_path = datapath + "_GT/SEG/" # man_seg01.tif
            
            for name in files:
            
                basename = os.path.basename(name)    
                targetname = ann_path + "man_seg" + basename[1:]
                print"target %s found" %targetname
                if not os.path.isfile(targetname):
                    print "Target file %s not found." % targetname
                    continue 
                
                
                image=parse_tif(name)
                image=np.mean(image, axis = 0)
                
                print "Image size is %s" % str(image.shape)
                #image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (128,128))       
                image = image.astype(np.float32) #sequential model is working with float32
                image -= np.min(image)
                image /= np.max(image)
                image = image[np.newaxis, :, :]
                X.append(image)
                labels.append(i)
            #    print "%d images collected..." % len(X)
                
                target=parse_tif(targetname)   
                print "Target size is %s" % str(target.shape)                
                target=np.any(target, axis = 0).astype(np.uint8)
                #target = cv2.imread(targetname,cv2.IMREAD_GRAYSCALE)
                target = cv2.resize(target, (target_dim,target_dim))
                target = target.astype(np.float32)
                target = np.clip(target, 0, 1)
                target = target[np.newaxis, ...]
                
                y.append(target)

                plt.figure(1)
                plt.imshow(np.concatenate((image[0,...], target[0,...]), axis = 1))
                plt.show()
                
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
                
                print "Dataset %s: %d/%d images done..." % (dataset, i, len(files))
            
        X = np.array(X)
        y = np.array(y)
        
        if not os.path.isdir(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))
        
        with h5py.File(cache_file, "w") as fp:
            fp["X"] = X
            fp["y"] = y
            fp["labels"] = labels
            
    w, h = 3,3
    model = Sequential()
    model.add(Convolution2D(32, w, h, border_mode='same',input_shape=(1,128,128)))
    #model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
     
    model.add(Convolution2D(32, w, h, border_mode='same'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
#    model.add(Convolution2D(32, w, h, border_mode='same'))
#    model.add(Activation('relu'))
    model.add(Convolution2D(1, w, h, border_mode='same'))
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
    
    for epoch in range(500):
        model.fit(X_train, y_train, batch_size=32, nb_epoch=1)
        y_hat = model.predict(X_test)
        test_error = np.mean(np.abs(y_hat - y_test))
        y_hat = model.predict(X_train)
        train_error = np.mean(np.abs(y_hat - y_train))
        
        training_errors.append(train_error)
        test_errors.append(test_error)
        print "Training error: %.4f" % train_error
        print "Test error: %.4f" % test_error
        
        if len(training_errors) > 10 and np.min(training_errors[:-10]) == np.min(training_errors):
            print "No improvement for 10 epochs. Stopping..."
            break            
    
    # Let's prepare illustration of how well the network segments
    
    y_hat = model.predict(X_test)
    for k in range(X_test.shape[0]):
        image = X_test[k,0,...]
        target = y_test[k, 0, ...]
        prediction = y_hat[k, 0, ...]
    
        image = (255 * normalize_image(image)).astype(np.uint8)
        target = (255 * normalize_image(target)).astype(np.uint8)   
        prediction = (255 * normalize_image(prediction)).astype(np.uint8)
        
        cv2.imwrite("result_images/test_img_%d.png" % k, image)
        cv2.imwrite("result_images/test_gt_%d.png" % k, target)
        cv2.imwrite("result_images/test_pred_%d.png" % k, prediction)

    y_hat = model.predict(X_train)
    for k in range(X_train.shape[0]):
        image = X_train[k,0,...]
        target = y_train[k, 0, ...]
        prediction = y_hat[k, 0, ...]
    
        image = (255 * normalize_image(image)).astype(np.uint8)
        target = (255 * normalize_image(target)).astype(np.uint8)   
        prediction = (255 * normalize_image(prediction)).astype(np.uint8)
        
        cv2.imwrite("result_images/train_img_%d.png" % k, image)
        cv2.imwrite("result_images/train_gt_%d.png" % k, target)
        cv2.imwrite("result_images/train_pred_%d.png" % k, prediction)
        
    plt.figure()
    
    plt.plot(training_errors, 'r-o', label = "Training error")
    plt.plot(test_errors, 'b-o', label = "Test error")
    plt.legend()
    plt.savefig("FinalResults.pdf")
   # plt.show()
        
    print "Training finished..."
     
