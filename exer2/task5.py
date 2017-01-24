# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:10:56 2017

@author: aliTakin
"""
#import matplotlib.pyplot as plt
#import numpy as np
#import cmath
#w = np.sqrt(0.25) * np.random.randn(100)
#x=[]
#n=np.arange(100)
#y=2*pi*(0.017)*n
#
#x=sin(y)+w
#plt.plot(x)
#
#
#scores = []
#frequencies = []
#for f in numpy.linspace(0, 0.5, 1000):
#        
## Create vector e. Assume data is in x.
#    n = np.arange(100)
#    oneImag=cmath.sqrt(-1)
#    z = -2*pi*oneImag*f*n# <compute -2*pi*i*f*n with i = sqrt(-1)>
#    e = np.exp(z)
#    score =np.abs(np.dot(x,e)) # <compute abs of dot product of x and e>
#    scores.append(score)
#    frequencies.append(f)
#fHat = frequencies[np.argmax(scores)]

import numpy as np
import matplotlib.pyplot as plt
import cmath

N = list(range(100))
X = []


for n in N:
    X.append(np.sin(2 * np.pi * 0.017 * n))

x = X + np.sqrt(0.25)*np.random.randn(100)

print(N)
print(X.__sizeof__())
print(x.size)

plt.plot(N,x)
#plt.show()


scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):

    nn = np.arange(100)
    z = -2*np.pi * cmath.sqrt(-1) * f * nn
    e = np.exp(z)
    score = np.abs(np.dot(e,x))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]
print(fHat)
