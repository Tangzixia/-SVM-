'''
sklearn中preprocessing模块，
主要针对数据的正则化，归一化，放缩进行处理，使得数据更好地被进行处理
'''

#coding=utf-8

from sklearn import svm
from sklearn import preprocessing as pre
import numpy as np

X_train = np.array([[ 1., -1.,  2.],
                     [ 2.,  0.,  0.],
                     [ 0.,  1., -1.]])


'''
#标准化
#即使得每一维的数据分布符合μ=0，δ=1的正态分布



X_scaled=pre.scale(X_train)
print(X_scaled)

scaler=pre.StandardScaler().fit(X_train)
print(scaler.mean_)
print(scaler.scale_)
print(scaler.transform(X_train))
'''


'''
#将矩阵放缩到一个范围  [min,max]
#缩放空间 ((x-x_min)/(x_max-x_min))//(max-_min)+min

#[0,1]缩放空间 (x-x_min)/(x_max-x_min)
min_max_x=pre.MinMaxScaler()
X_train_minmax = min_max_x.fit_transform(X_train)
print(X_train_minmax)
x_test=np.array([[-3,-1,4]])
x_test_minmax=min_max_x.transform(x_test)
# (x_test-x_min)/(x_max-x_min)
print(x_test_minmax)

#[-1,1]  x/x_max
max_abs_x=pre.MaxAbsScaler()
x_train_maxabs=max_abs_x.fit_transform(X_train)
print(x_train_maxabs)
#x_test/x_max 注意这儿的x_max是X_train中的对饮的每一个维度的X_max
x_test=np.array([[-3,-1,4]])
x_test_maxabs=max_abs_x.transform(x_test)
print(x_test_maxabs)
'''

#正则化 normalize，注意正则化是对每个样本的不同维进行操作，使得其p次方的和为1
#xi=(xi0,xi1,xi2,xi3,...,xin)
#xi_l1=(xi0,xi1,xi2,xi3,...,xin)//∑|xi_j|(j=1..n)
#xi_l2=(xi0,xi1,xi2,xi3,...,xin)//sqrt(∑(xi_j)^2) (j=1..n)
#X_normalized=pre.normalize(X_train,norm="l1")
#print(X_normalized)
normalizer=pre.Normalizer().fit(X_train)
X_normalized=normalizer.transform(X_train)
print(X_normalized)
