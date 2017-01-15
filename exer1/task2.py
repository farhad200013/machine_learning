# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:02:54 2017

@author: aliTakin
"""

#this is task three

import numpy as np 
from scipy.ndimage.morphology import white_tophat
import matplotlib.pyplot as plt 

import matplotlib.image as img

temp=img.imread(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\oulu.jpg")
plt.imshow(temp)  
print(np.shape(temp))
tempR=temp[:,:,0]
tempG=temp[:,:,1]
tempB=temp[:,:,2]

mean_whole_image=np.mean(temp)
meanR=np.mean(tempR,axis=0)
meanG=np.mean(tempG,axis=0)
meanB=np.mean(tempB,axis=0)

plt.figure()
plt.imshow(white_tophat(temp, size = 10))  

