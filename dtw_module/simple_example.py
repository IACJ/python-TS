# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:15:25 2017

@author: IACJ
"""


import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw


np.set_printoptions(threshold=999)  #全部输出

x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

plt.plot(x)
plt.plot(y)
plt.show()
dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))


print ('Minimum distance found:', dist)

print('cost:\n',cost)

print('acc:\n',acc)

print('path:\n',path)

plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))
