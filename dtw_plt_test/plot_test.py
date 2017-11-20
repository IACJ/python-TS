# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:58:40 2017

@author: IACJ
"""

print('begin...')

import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw

#plt.plot([1, 2, 3, 4])
#plt.ylabel('some numbers')
#plt.show()
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#plt.show()
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()

if __name__ == '__main__':

    while True:
        from sklearn.metrics.pairwise import euclidean_distances
        
        n=30
        x=[]
        for i in range(n):
            x.append(np.random.randint(7))
        y = x *3
        
        x = np.array(x)
        y = np.array(y)
        
        plt.plot(x-7,'r-')
        plt.plot(y)
        plt.show()
        
        dist_fun = euclidean_distances
        dist, cost, acc, path = dtw(x, y, dist_fun)
        
        
        plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
        plt.plot(path[0], path[1], '-ro') # relation
        plt.axis('tight')
        plt.show()
        
        x = x-7
        plt.plot(x,'r-')
        plt.plot(y)
        for i in range(len(path[1])):
            plt.plot([path[0][i], path[1][i]],  [x[path[0][i] ],  y[path[1][i]]],'y')
        plt.show()
        
        print("DTW = ",acc[-1][-1])

#    # vizualize
#    plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
#    plt.plot(path[0], path[1], '-o') # relation
##    plt.xticks(range(len(x)), x)
##    plt.yticks(range(len(y)), y)
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.axis('tight')
#    plt.title('Minimum distance: {}'.format(dist))
#    plt.show()
    
    ####################################################
#    x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0])
#    z =  np.array(list(y) *3)
#    
#
#    
#    x = x - 7
#    plt.plot(x,'r-')
#    plt.plot(z)
#    plt.savefig('out.png')
#    plt.show()
#
#    dist, cost, acc, path = dtw(x, z, dist_fun)
#    
#    plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
#    plt.plot(path[0], path[1], '-ro') # relation
#    plt.axis('tight')
#    plt.savefig('out1.png')
#    plt.show()
#    
#
#    
#    
#    
#    plt.savefig('out2.png')
#    
#    
#    plt.show()

