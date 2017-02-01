# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:03:15 2017

@author: aliTakin
"""

import numpy as np
import matplotlib.pyplot as plt 

nn=np.arange(500,601)
nz=np.zeros(500)
nz2=np.zeros(300)

nc=np.concatenate((nz,np.cos(2*np.pi*0.1*nn),nz2),axis=0)

y_n= nc + np.sqrt(0.5) * np.random.randn(nc.size)



y = np.convolve(nc, y_n, 'same')


plt.figure(1)
plt.subplot(311)        
plt.plot(nc)

plt.subplot(312)
plt.plot(y_n)

plt.subplot(313)
plt.plot(y)
plt.show()