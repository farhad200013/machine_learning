# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 20:49:58 2017

@author: aliTakin
"""
import cmath

scores = []
frequencies = []
for f in numpy.linspace(0, 0.5, 1000):
# Create vector e. Assume data is in x.
n = numpy.arange(100)
z = # <compute -2*pi*i*f*n with i = sqrt(-1)>
e = numpy.exp(z)
score = # <compute abs of dot product of x and e>
scores.append(score)
frequencies.append(f)
fHat = frequencies[np.argmax(scores)]