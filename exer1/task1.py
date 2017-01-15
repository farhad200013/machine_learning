# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 10:59:56 2017

@author: aliTakin
"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


loaded=np.loadtxt(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv")
size=np.shape(loaded)
plt.plot(loaded[:,0],loaded[:,1])

ax = plt.subplot(1, 1, 1, projection = "3d")

plt.plot(loaded[:,0],loaded[:,1], loaded[:,2])