# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:28:57 2017

@author: IACJ
"""

from os import path
from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
from dtw import dtw
from sklearn.metrics.pairwise import euclidean_distances


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:55:45 2017

@author: IACJ
"""

import numpy as np
from numpy import *
import sys
import math
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

set_printoptions(threshold=100)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

class GT(object):
    'GenoType，基因型'
    def __init__(self,randomInit=True):
        '随机初始化基因型的各个基因'

        if (randomInit):
            self.gene=np.zeros(15,dtype=int)
            for i in range(5):
                self.gene[i] = random.randint(1,3)
            x = list([1,2,3,4,5,1,2,3,4,5])
            random.shuffle(x)
            self.gene[5:15] = x
        else :
            self.gene=np.zeros(15,dtype=int)
            
    def selfXover(self):
        '自交函数，自我进化'
        first = random.randint(5,14)
        second = random.randint(5,14)
        self.gene[first],self.gene[second] = self.gene[second],self.gene[first]
        
        P = random.randint(0,4)
        L = random.randint(1,3)
        self.gene[P] = L

    def copy(self):
        '深复制函数'
        copy = GT(randomInit=False)
        copy.gene = self.gene.copy()
        return copy
    def show(self):
        '打印该基因型'
        print(self.gene[0:5],self.gene[5:15])
      

class GA(object):
    'Genetic Algorithm, 遗传算法'
    def __init__(self, data,popSize=200, maxGens=300,\
                 pXover=0.3,pMutation=0.02,report_detail=False):
        '构造函数：记录各个参数'
        self.data = data
        self.popSize = popSize
        self.maxGens = maxGens
        self.pXover = pXover
        self.pMutation = pMutation
        self.report_detail=report_detail      
        
    def run(self):
        'GA的运行入口'
        self.initialize()
        self.evaluate()
        self.elitist()
        while (self.generation<self.maxGens):
            self.generation += 1
            self.select()
            self.crossover()
            self.mutate()
            self.evaluate()
            self.elitist()
            
        return self.bestGT

        
    def initialize(self):
        '初始化'
        
        self.generation=0
        self.bestCurve = []
        self.avgCurve = []
        self.worstCurve = []
        
        self.GTs=[]
        for i in range(self.popSize):
            self.GTs.append(GT(randomInit=True))
        self.fitness = np.zeros(self.popSize)
            
        self.bestGT = GT(randomInit=False)
        self.bestFitness = -9999
                 
    def evaluate(self):
        '评估fitness'
        for i,gt in enumerate(self.GTs):
            self.fitness[i] = 0 
            for j in range(10):              
                S = j+1
                P = gt.gene[5+j]
                L = gt.gene[P-1]              
                loc1 = 'P'+str(P)+'.L'+ str(L)
                loc2 = 'S'+str(S)
                v = self.data.loc[loc1,loc2]
                self.fitness[i] -=v

        
    
    def select(self):
        '选择算子: 数据归一化 + softmax + 轮盘赌'
        
        t_fitness = self.fitness.copy()
        t_mean = t_fitness.mean()
        t_max = t_fitness.max()
        t_min = t_fitness.min()
        t_fitness = (t_fitness-t_mean)/ (t_max - t_min +1)
        t_fitness = softmax(t_fitness)
        
        c_fitness = t_fitness.cumsum()    

        
        newGTs = []
        for i in range(self.popSize):
            p = random.random()
            for j in range(self.popSize):
                if p < c_fitness[j]:
                    newGTs.append(self.GTs[j].copy())
                    break       
        self.GTs = newGTs

    
    def crossover(self):
        '交配算子:自交'
        mem = 0
        first = 0
        one = 0   
        for i in range(self.popSize):
            p = random.random()
            if p < self.pXover :
                self.GTs[i].selfXover()

    def mutate(self):
        '基因变异:重生'
        for i in range(self.popSize):
            p = random.random()
            if p < self.pMutation:
                self.GTs[i] = GT(randomInit=True)
    
    
    def elitist(self):
        '每代保留最优，取代最差'
            
            
        if self.fitness.max() > self.bestFitness:
            self.bestGT = self.GTs[self.fitness.argmax()].copy()
            self.bestFitness = self.fitness.max()
        
        self.bestCurve.append(self.bestFitness)
        self.avgCurve.append(self.fitness.mean())
        self.worstCurve .append(self.fitness.min())

        self.GTs[self.fitness.argmin()] = self.bestGT.copy()
    
    def showAllGT(self):
        '打印所有存活的基因型'
        
        print(self.generation,self.bestFitness)
        for i in range(self.popSize):
            self.GTs[i].show()
    def showCurve(self):
        '展示进化曲线'
        
        plt.plot(self.bestCurve)
        plt.plot(self.avgCurve)
        plt.plot(self.worstCurve)
        plt.show()
        plt.savefig('./output.png')
                         
            


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



    
    
def show(x,y,path,step=1,off=0):
    plt.imshow(cost.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    plt.plot(path[0], path[1], '-ro') # relation
    plt.axis('tight')
    plt.show()
    
    x = x-10
    xIndex = np.zeros(len(x))
    for i in range(len(x)):
        xIndex[i] = i;
        
    xIndex +=off
    
    plt.plot(xIndex,x,'r-')
    plt.plot(y)
    for i in range(0,len(path[1]),step):
        plt.plot([path[0][i]+off, path[1][i]+off],  [x[path[0][i]],  y[path[1][i]+off]],'y')
    plt.savefig("out.png")
    plt.show()
    



if __name__ == '__main__':
    print('hello')
    
#    ucr_dataset_base_folder = expanduser('~/UCR_TS_Archive_2015')
#    ucr_dataset_name = 'Gun_Point'
#    
#    train_data, train_labels, test_data, test_labels = load_dataset(ucr_dataset_name,ucr_dataset_base_folder)
#
#
#    x = train_data[1]
#    y = np.tile(x,3)

    n=10
    x=[]
    
    for i in range(n):
        x.append(np.random.randint(10))
    y = x *3
    for i in range(len(y)):
        y[i] += random.randint(-2,2)
    
    x = np.array(x)
    y = np.array(y)
    
 
    bestP=-1
    bestDTW=999
    
    for window in range(1,len(y)+1):
        print()
        print("window=",window)
        print()
        for i in range(len(y)-window+1):
            dist, cost, acc, path = dtw(x, y[i:i+window], euclidean_distances)
            if acc[-1][-1] <= bestDTW:
                bestDTW = acc[-1][-1]
                bestP = i;
        print(bestP)
        
        show(x,y,path,1,bestP)
        print("DTW = ",bestDTW)
    
    print ('BEGIN')
    btime = time.time()   
    ############# 使用默认参数 #######################
#    testGA = GA(data)
#    testGA.run()
#    testGA.showCurve()
#    print(testGA.bestFitness)
#    testGA.bestGT.show()
    ###################################################
    etime = time.time()
#    print("耗时：",etime - btime )
    print ('END.')
    
    
    
######################################################   
#    for i in range(150):
#        if test_labels[i]>0.5:
#            plt.plot(test_data[i],'r')
#        else:
#            plt.plot(test_data[i],'b')
#    plt.savefig('test.png')
#    plt.show()
#    
#    for i in range(50):
#        if train_labels[i]>0.5:
#            plt.plot(train_data[i],'r')
#        else:
#            plt.plot(train_data[i],'b')
#    plt.savefig('train.png')
#    plt.show()
######################################################
    
    
    
    
    