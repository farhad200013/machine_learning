# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:50:30 2017

@author: aliTakin
"""


import numpy as np
import matplotlib.pyplot as plt 

nn=np.arange(500,601)
nz=np.zeros(500)
nz2=np.zeros(300)

nc=np.concatenate((nz,np.cos(2*np.pi*0.03*nn),nz2),axis=0)
#plt.plot(nc)

y_n= nc + np.sqrt(0.5) * np.random.randn(nc.size)

#plt.plot(y_n)
nn2=(np.arange(100))
h = np.exp(-2 * np.pi * 1j * 0.03 *nn2 )
y = np.abs(np.convolve(y_n,h,'same'))

#plt.plot(y)

plt.figure(1)
plt.subplot(311)
plt.plot(nc)

plt.subplot(312)
plt.plot(y_n)

plt.subplot(313)
plt.plot(y)
plt.show()
