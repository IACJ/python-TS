# Python - Time Series

嗯，造轮子总是无法避免的，让我们开心的造轮子吧！

## Naive_TS_Similarity

Outline:
~~~python
class Naive_TS_Similarity(object):
'''
一个Naive的时间序列相似性测量类，
包括欧氏距离(ED)、DTW距离和Z-Normalization。
为了体现Naive的一面，特地没有使用向量化实现。
'''
    def ED(a,b):
    '''
    接收参数：两个等长的时间序列
    返回结果：它们之间的欧氏距离(ED)
    '''
    def DTW(a, b, print_detail=True):
     '''
    接收参数：两个时间序列，以及是否打印详情
    返回结果：它们之间的DTW距离
    '''
    def z_normalization(a):
    '''
    接收参数：一个时间序列
    返回结果：正态化后的时间序列，旧均值，旧方差
    '''
~~~ 

## GA