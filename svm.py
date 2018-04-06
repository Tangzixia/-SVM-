#coding=utf-8
#https://www.cnblogs.com/xiaotan-code/p/6680438.html
import os
import math
import collections
import numpy as np
from sklearn import svm
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

def Euclidean(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())

def fwalker(path):
    fileArray = []
    for root, dirs, files in os.walk(path):
        for fn in files:
        	eachpath = str(root+'\\'+fn)
        	fileArray.append(eachpath)
    return fileArray

def orderdic(dic, reverse):
    ordered_list = sorted(
        dic.items(), key=lambda item: item[1], reverse=reverse)
    return ordered_list

def get_data(data_path):
    label_vec = []
    files = fwalker(data_path)
    for file in files:
        ech_label_vec = []
        ech_label = int((file.split('\\'))[-1][0])# 获取每个向量的标签
        ech_vec = ((np.loadtxt(file)).ravel())# 获取每个文件的向量
        ech_label_vec.append(ech_label) # 将一个文件夹的标签和向量放到同一个list内
        ech_label_vec.append(ech_vec) # 将一个文件夹的标签和向量放到同一个list内，目的是将标签和向量对应起来，类似于字典，这里不直接用字典因为字典的键（key）不可重复。
        label_vec.append(ech_label_vec) # 再将所有的标签和向量存入一个list内，构成二维数组
    return label_vec


def find_label(train_vec_list, vec, k):
    get_label_list = []
    for ech_trainlabel_vec in train_vec_list:
        ech_label_distance = []
        train_label, train_vec = ech_trainlabel_vec[0], ech_trainlabel_vec[1]
        vec_distance = Euclidean(train_vec, vec)# 计算距离
        ech_label_distance.append(train_label)
        ech_label_distance.append(vec_distance)# 将距离和标签对应存入list
        get_label_list.append(ech_label_distance)
    result_k = np.array(get_label_list)
    order_distance = (result_k.T)[1].argsort()# 对距离进行排序
    order = np.array((result_k[order_distance].T)[0])
    top_k = np.array(order[:k], dtype=int) # 获取前k距离和标签
    find_label = orderdic(collections.Counter(top_k), True)[0][0]# 统计在前k排名中标签出现频次
    return find_label


def classify(train_vec_list, test_vec_list, k):
    error_counter = 0 #计数器，计算错误率
    for ech_label_vec in test_vec_list:
        label, vec = ech_label_vec[0], ech_label_vec[1]
        get_label = find_label(train_vec_list, vec, k) # 获得学习得到的标签
        print('Original label is:'+str(label) +
              ', kNN label is:'+str(get_label))
        if str(label) != str(get_label):
            error_counter += 1
        else:
            continue
    true_probability = str(round((1-error_counter/len(test_vec_list))*100, 2))+'%'
    print('Correct probability:'+true_probability)


def main():
    k = 3
    train_data_path =".\CCPP\\digits\\trainingDigits"
    test_data_path =".\Users\\炜依\\Desktop\\test_code\\CCPP\\digits\\trainingDigits"
    train_vec_list = get_data(train_data_path)
    test_vec_list = get_data(test_data_path)
    classify(train_vec_list, test_vec_list, k)

if __name__ == '__main__':
    train_data_path ="C:\\Users\\炜依\\Desktop\\test_code\\CCPP\\digits\\trainingDigits"
    test_data_path ="C:\\Users\\炜依\\Desktop\\test_code\\CCPP\\digits\\trainingDigits"
    train_vec_list = get_data(train_data_path)
    test_vec_list = get_data(test_data_path)
    
    x_train=[]
    y_train=[]
    for item in train_vec_list:
    	x_train.append(item[1])
    	y_train.append(item[0])
    print(x_train[0])
    x_test=[]
    y_test=[]
    for item in test_vec_list:
    	x_test.append(item[1])
    	y_test.append(item[0])

    #ovo代表的是一对一，ovr代表的是一对多，
    #svm_=svm.SVC(decision_function_shape="ovo")  #(1934,45) n*(n-1)/2
    svm_=svm.SVC(decision_function_shape="ovr")  #(1934,10) n
    #x_train=preprocessing.scale(x_train)
    x_train=preprocessing.normalize(x_train,norm="l2")

    svm_.fit(x_train,y_train)

    a_l=[]
    for j in range(len(x_train)):
        l=[]
        for i in range(32):
            l.append(i)
        a_l.append(l)
    dec=svm_.decision_function(a_l)
    print(dec.shape)
    pre_s=svm_.predict(x_test)
    #print(svm_.support_vectors_)
    #print(svm_.support_)
    #print(svm_.n_support_)
    '''
    count=0
    for i in range(len(pre_s)):
        if(pre_s[i]==y_test[i]):
            count=count+1
        else:
            continue
    print("准确率："+str(count//len(x_test)))
    '''
    #查看在测试集上的准确率
    print(svm_.score(x_test,y_test))


    
