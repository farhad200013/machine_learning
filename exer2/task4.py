# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 19:23:51 2017

@author: aliTakin
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
def gaussian(x,mu,sigma):
        
        return(1/math.sqrt(2*pi*(sigma)**2)*exp((-1/2*sigma**2)*(x-mu)**2 )) 

def log_gaussian(x,mu,sigma):
        
        return(exp(1/math.sqrt(2*pi*(sigma)**2)+((-(x-mu)**2 )/2*sigma**2))) 


p = gaussian(np.linspace(-5.0, 5.0, num=100), 0, 1)
plt.plot(np.linspace(-5.0, 5.0, num=100),p)
pLog = log_gaussian(np.linspace(-5.0, 5.0, num=100), 0, 1)
plt.plot(np.linspace(-5.0, 5.0, num=100),pLog)
