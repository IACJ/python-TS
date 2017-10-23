# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 17:55:45 2017

@author: IACJ
"""

import numpy as np
import sys
import math
import random
import time
import pandas as pd

class GT(object):
    'GenoType，基因型'
    def __init__(self,nVars,varBound):
        '根据数值范围随机初始化各个基因'
        self.nVars = nVars
        self.varBound = varBound
        self.fitness = 0
        self.gene=np.zeros(self.nVars)
        self.upper=[0] * self.nVars
        self.lower=[0] * self.nVars
        for i in range(self.nVars):
            self.upper[i], self.lower[i] = self.varBound[i]
            self.gene[i] = (self.upper[i]-self.lower[i])*random.random() +self.lower[i]
            
    def copy(self):
        '深复制函数'
        copy = GT(self.nVars,self.varBound)
        copy.fitness = self.fitness
        copy.gene = self.gene.copy()
        return copy
    
      

class GA(object):
    'Genetic Algorithm, 遗传算法'
    def __init__(self, popSize=100, maxGens=500,\
                 varBound=None,pXover=0.7,pMutation=0.07,report_detail=False):
        if varBound==None:
            print("varBound is required")
            assert(varBound!=None)

        self.popSize = popSize
        self.maxGens = maxGens
        self.nVars = len(varBound)
        self.varBound = varBound
        self.pXover = pXover
        self.pMutation = pMutation
        self.report_detail=report_detail
        
    def describe(self):
        print("popSize:",self.popSize)
        print("maxGens:",self.maxGens)
        print("nVars:",self.nVars)
        print("report_detail:",self.report_detail)
        
    def run(self):
        'GA的运行入口'
        self.generation=0
        self.initialize()
        self.evaluate()
        self.keep_the_best()
        while (self.generation<self.maxGens):
            self.generation += 1
            self.select()
            self.crossover()
            self.mutate()
            self.report()
            self.evaluate()
            self.elitist()
        return self.bestGT

        
    def initialize(self):
        '初始化'
        self.population = [GT(self.nVars,varBound)] * self.popSize
        self.bestGT = GT(self.nVars,varBound)
         
    def evaluate(self):
        '评估fitness'
        for i in self.population:
            i.fitness = 21.5 + \
                i.gene[0]*math.sin(4*math.pi*i.gene[0]) + \
                i.gene[1]*math.sin(20*math.pi*i.gene[1])
    
    def keep_the_best(self):
        '保存最优GT'
        for i in self.population:
            if i.fitness > self.bestGT.fitness:
                self.bestGT = i.copy()
                
        
    
    def select(self):
        '选择算子--轮盘赌'
        f_sum = 0
        t_fitness = np.zeros(self.popSize) # true fitness
        r_fitness = np.zeros(self.popSize) # relative fitness
        c_fitness = np.zeros(self.popSize) # cumulative fitness  
        for i,j in enumerate(self.population):
            t_fitness[i] = j.fitness
        
        f_sum = t_fitness.sum()
        r_fitness = t_fitness / f_sum
        c_fitness = r_fitness.cumsum()
        
        newPopulation = []
        for i in range(self.popSize):
            p = random.random()
            for j in range(self.popSize):
                if (p < c_fitness[j]) :
                    newPopulation.append(self.population[j].copy())
                    break       
        self.population = newPopulation

    
    def crossover(self):
        '交配算子'
        mem = 0
        first = 0
        one = 0   
        for i in range(self.popSize):
            p = random.random()
            if p < self.pXover :
                first += 1
                if first % 2 ==0:
                    self.Xover(one,i)
                else:
                    one = i
    
    def Xover(self,GT1,GT2):
        '基因交配'
        if (self.nVars ==2):
            point = 1
        else:
            point = random.randrange(1,self.nVars); 
        for i in range(point):
            self.population[GT1].gene[i],self.population[GT2].gene[i] = \
                self.population[GT2].gene[i],self.population[GT1].gene[i]
        
    
    def mutate(self):
        '基因变异'
        for i in range(self.popSize):
            for j in range(self.nVars):
                p = random.random()
                if p < self.pMutation:
                    self.population[i].gene[j] = \
                    (self.population[i].upper[j]-self.population[i].lower[j])*random.random() +self.population[i].lower[j]
    
    def report(self):
        '打印详情'
        if (self.report_detail) :
            print(self.generation,':',self.bestGT.fitness)
    
    def elitist(self):
        '每代保留最优，取代最差'
        t_fitness = np.zeros(self.popSize) # true fitness
        for i,j in enumerate(self.population):
            t_fitness[i] = j.fitness
        self.bestGT = self.population[t_fitness.argmax()].copy()
        self.population[t_fitness.argmin()] = self.bestGT.copy()
            
                     
            
if __name__ == '__main__':
    print ('BEGIN')
    varBound=[(-3.0,12.1),(4.1,5.8)]
    btime = time.time()
    
    result = np.zeros((9,9))
    df = pd.DataFrame(result,index=np.arange(0.01,0.1,0.01),columns=np.arange(0.1,1.0,0.1))
    
    for pMutation in np.arange(0.01,0.1,0.01):
        for pXover in np.arange(0.1,1.0,0.1):
            print(pMutation,pXover)    
            temp_sum = 0
            for i in range(10):   
                myGA = GA(varBound=varBound,pXover=pXover,pMutation=pMutation)
                bestGT = myGA.run()
                temp_sum += bestGT.fitness
                print(bestGT.fitness)
            temp_sum /= 10                 
            df.at[pMutation,pXover] = temp_sum
            print ('pMutation',pMutation,'pXover',pXover,'bestGT.fitness',temp_sum)
    
    df.plot()
    df.T.plot()
    df.to_csv('result2.csv')
    etime = time.time()
    print(etime - btime )
    print ('END.')