# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 10:31:00 2017

@author: IACJ
"""

from os import path
from os.path import expanduser
from numpy import genfromtxt
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from numpy import array, zeros, argmin, inf

# 别人的DTW代码，直接拷贝进来，省去import和pip install的麻烦
def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path
# 别人的DTW代码，直接拷贝进来，省去import和pip install的麻烦
def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)



# 加载 UCR 数据集的函数
def load_dataset(dataset_name, dataset_folder):
    dataset_path = path.join(dataset_folder, dataset_name)
    train_file_path = path.join(dataset_path, '{}_TRAIN'.format(dataset_name))
    test_file_path = path.join(dataset_path, '{}_TEST'.format(dataset_name))

    # training data
    train_raw_arr = genfromtxt(train_file_path, delimiter=',')
    train_data = train_raw_arr[:, 1:]
    train_labels = train_raw_arr[:, 0] - 1
    # one was subtracted to change the labels to 0 and 1 instead of 1 and 2

    # test_data
    test_raw_arr = genfromtxt(test_file_path, delimiter=',')
    test_data = test_raw_arr[:, 1:]
    test_labels = test_raw_arr[:, 0] - 1

    return train_data, train_labels, test_data, test_labels


# 绘图函数
def show(xx,yy,path,step=1):
    '用于绘图的函数'
    
    x = np.array(xx)
    y = np.array(yy)
    p0 = np.array(list(path[0]),dtype=int)
    p1 = np.array(list(path[1]),dtype=int)
    
    #绘制矩阵图
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    plt.plot(p0, p1, '-ro') # relation
    plt.axis('tight')
    plt.show()
    
    # 绘制对齐图
    x = x-10
    xIndex = np.zeros(len(x))
    for i in range(len(x)):
        xIndex[i] = i;
        
    
    plt.plot(xIndex,x,'r-')
    plt.plot(y)

    #绘制对齐线
    for i in range(0,len(p1),step):
        plt.plot([p0[i], p1[i]],  [x[p0[i]],  y[ p1[i] ] ],'y')
    plt.show()
  
    
if __name__ == '__main__':    

    print("program begin ")
    
########### 使用 UCR 数据集 ###############
#    需要先下载 UCR 数据集才能使用
#    ucr_dataset_base_folder = expanduser('~/UCR_TS_Archive_2015')
#    ucr_dataset_name = 'Gun_Point'    
#    train_data, train_labels, test_data, test_labels = load_dataset(ucr_dataset_name,ucr_dataset_base_folder)
#    x = train_data[1]
########### 使用 UCR 数据集 ###############    


########### 使用 随机生成数据  ###############    
    n=10
    x=[]
    for i in range(n):
        x.append(np.random.randint(6))
    x = np.array(x)
########### 使用 随机生成数据  ###############   
    
    DTWCurve = []
    minDTWCurve = []
    index=[]
    for i in range(1,10):
        y = np.tile(x,i)

        dist, cost, acc, pathh = dtw(x, y, euclidean_distances)
        show(x,y,pathh)
        print('DTW = ',acc[-1][-1])
 
        DTWCurve.append(acc[-1][-1])
        minDTWCurve.append(dist)
        index.append(i)
        
        
    plt.plot(index,DTWCurve)
    plt.show()
    plt.plot(index,minDTWCurve)
    plt.show()
#    plt.savefig('curve.png')
           
    
        
       
    