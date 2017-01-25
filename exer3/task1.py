# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 11:03:15 2017

@author: aliTakin
"""

import numpy as np
import matplotlib.pyplot as plt 

nn=np.arange(500,601)
nz=np.zeros(409)
nz2=np.zeros(301)

nc=np.concatenate((nz,np.cos(2*np.pi*0.1*nn),nz2),axis=0)
#plt.plot(nc)

y_n= nc + np.sqrt(0.5) * np.random.randn(nc.size)

#plt.plot(y_n)

y = np.convolve(nc, y_n, 'same')

plt.plot(y)