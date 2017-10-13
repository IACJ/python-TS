# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 23:32:20 2017

@author: IACJ
"""

import numpy as np
import math


class Naive_TS_Similarity(object):
    '''
    一个Naive的时间序列相似性测量类，
    包括欧氏距离(ED)、DTW距离、Z-Normalization。
    为了体现Naive的一面，
    特地没有使用向量化实现。
    '''
            
    def ED(a,b):
        '''
        接收参数：两个等长的时间序列
        返回结果：它们之间的欧氏距离(ED)
        备注：没有对结果取平方根，因为不影响比较大小
        过程说明：
            过程1，检查参数
            过程2，累加计算欧氏距离(ED)
            过程3，返回结果-欧氏距离(ED)
        '''
        
        # 过程1，检查参数
        if (a.shape!=b.shape):
            print('参数错误')
            return 0
       
        # 过程2，累加计算欧氏距离(ED)
        dis = 0
        for i in range(len(a)):
            dis  += (a[i]-b[i]) ** 2
            
        # 过程3，返回结果-欧氏距离(ED)
        return dis 
    
    def DTW(a, b, print_detail=True):
        '''
        接收参数：两个等长的时间序列，以及是否打印详情
        返回结果：它们之间的DTW距离
        备注：没有对结果取平方根，因为不影响比较大小
        过程说明：
            过程1，构建 D 矩阵
            （可选过程）打印 D 矩阵
            过程2，利用动态规划(DP)计算扭曲路径
            （可选过程）打印扭曲路径
            （可选过程）打印 DP 矩阵
            过程3，返回结果-DTW距离
        '''
        
        # 过程1，构建 D 矩阵
        D = np.zeros((len(a),len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                D[i][j] = (a[i]-b[j]) ** 2
        # （可选过程）打印 D 矩阵
        if print_detail:
            print("D矩阵:")
            print(D)
           
        # 过程2，利用动态规划(DP)计算扭曲路径
        # 其中pointer用来记录路径，可以先不看
        dis = 0 
        DP = np.zeros((len(a),len(b)))
        pointer = np.zeros((len(a),len(b),2))
        
        for i in range(len(a)):
            for j in range(len(b)):
                
                min_from = float('inf')
 
                if (j==0)and(i==0) :
                    min_from = 0
                    pointer[i][j][0],pointer[i][j][1] = (-1,-1)
    
                if (i > 0)and(DP[i-1][j] < min_from) :
                    min_from = DP[i-1][j]
                    pointer[i][j][0],pointer[i][j][1] = (i-1,j)
                    
                if (j > 0)and(DP[i][j-1] < min_from) :
                    min_from = DP[i][j-1]
                    pointer[i][j][0],pointer[i][j][1] = (i,j-1)   
                    
                if (j>0)and(i>0)and(DP[i-1][j-1] < min_from) :
                    min_from = DP[i-1][j-1]
                    pointer[i][j][0],pointer[i][j][1] = (i-1,j-1)          
                    
                DP[i][j] = D[i][j] + min_from
        # （可选过程）打印扭曲路径
        if print_detail:
            i,j = (len(a)-1,len(b)-1)
            
            path = np.zeros((len(a),len(b)))
            while ( (i,j) != (-1,-1) ):
                path[i][j] = 1
                i,j = int(pointer[i][j][0]),int(pointer[i][j][1])
            print ('扭曲路径如下：')
            print (path)
                
        # （可选过程）打印 DP 矩阵
        if print_detail:
            print("DP矩阵:")
            print(DP)
        
        # 过程3，返回结果-DTW距离
        return DP[len(a)-1][len(b)-1]
        
    def z_normalization(a):
        '''
        接收参数：一个时间序列
        返回结果：正态化后的时间序列，旧均值，旧方差
        过程说明：
            过程1，计算平均值mean
            过程2，计算方差variance、标准差stanDev
            过程3，计算正态化后的时间序列
            过程4，返回结果（正态化后的时间序列，旧均值，旧方差）
        '''
        
        # 过程1，计算平均值mean
        meanSum = 0
        for i in range(len(a)):
            meanSum += a[i]
        mean = meanSum / len(a)
        
        # 过程2，计算方差variance、标准差stanDev
        varianceSum = 0
        for i in range(len(a)):
            varianceSum += (a[i] - mean) **2
        variance = varianceSum / len(a)
        standard_deviation = math.sqrt(variance)

        # 过程3，计算正态化后的时间序列
        b = np.zeros(len(a))    
        for i in range(len(a)):
            b[i] = (a[i]-mean) / standard_deviation
        
        # 过程4，返回结果（正态化后的时间序列，旧均值，旧方差）
        return b, mean, variance
        
    
# 程序从这里开始运行
if __name__ == '__main__':
    print("Log:程序开始")  

    # 定义测试的数据
    test_a=np.array([1,2,-3,4,5,6])
    test_b=np.array([1,2,13,4,5,8])
    print("test_a序列：")
    print(test_a)
    print("test_b序列：")
    print(test_b)
    print()

    # 测试该类的三个静态方法
    ED_dis = Naive_TS_Similarity.ED(test_a,test_b)
    print('ED_dis: '+str(ED_dis))
    print()
    
    print('DTW :')
    DTW_dis = Naive_TS_Similarity.DTW(test_a,test_b)
    print('DTW_dis: '+str(DTW_dis))
    print()
    
    newTs,mean,variance = Naive_TS_Similarity.z_normalization(test_a)
    print('z-normalization:')
    print(newTs)
    print('old mean : '+str(mean))
    print('old variance : '+str(variance))
    print()
    
    print("Log:程序程序结束") 
    
    
        