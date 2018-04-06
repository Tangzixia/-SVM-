#coding=utf-8
#https://www.cnblogs.com/xiaotan-code/p/6680438.html
import os
import math
import collections
import numpy as np

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
    test_data_path =".\CCPP\\digits\\trainingDigits"
    train_vec_list = get_data(train_data_path)
    test_vec_list = get_data(test_data_path)
    classify(train_vec_list, test_vec_list, k)

if __name__ == '__main__':
    #main()
    #weights:1）uniform,等比重投票；2）distance，距离反比投票；3）callable，表示自己定义的一个函数，接受一个距离数组，返回一个权值数组
    #algorithm:1）auto:表示根据数据类型和结构选择合适算法；2）ball_tree；3）kd_tree；4）brute：暴力搜索
    #metric:距离的计算，默认是闵氏距离，p=2为欧式距离，p=1为曼哈顿距离
    #n_jobs:表示并发线程的数量
    knn=KNeighborsClassifier(n_neighbors=3,weights="uniform",algorithm="kd_tree",leaf_size=30,metric="minkowski",p=5,n_jobs=10)
    train_data_path ="C:\\Users\\炜依\\Desktop\\test_code\\CCPP\\digits\\trainingDigits"
    test_data_path ="C:\\Users\\炜依\\Desktop\\test_code\\CCPP\\digits\\trainingDigits"
    train_vec_list = get_data(train_data_path)
    test_vec_list = get_data(test_data_path)
    
    x_train=[]
    y_train=[]
    for item in train_vec_list:
    	x_train.append(item[1])
    	y_train.append(item[0])
    x_test=[]
    y_test=[]
    for item in test_vec_list:
    	x_test.append(item[1])
    	y_test.append(item[0])
    #x_test,y_test=test_vec_list[:,1],test_vec_list[:,0]
    knn.fit(x_train,y_train)

    prob=0
    #predict为预测函数，接收输入数组类型测试样本，一般是二维数组，行代表样本，列代表属性
    #输出：如果每个样本只有一个输出，则输出为一个一维数组；
    #		如果每个样本的输出是多维的，则输出二维数组，每一行是一个样本，每一列是一维输出
    y_test_pred=knn.predict(x_test)
    probability=knn.predict_proba(x_test)
    #print(probability)
    #kneighbors方法接收3个参数，X表示最近邻训练样本
    #n_neighbors表情需要寻找目标样本最进的几个最近邻的样本
    #return_distance表明是否需要返回具体的距离值
    neighborpoint=knn.kneighbors(x_test,5,False)
    print(neighborpoint)
    #score是计算准确率的函数，该函数不是KneighborsClassifier的方法
    #而是从它的父类继承下来的方法！
    score=knn.score(x_test,y_test,sample_weight=None)
    print(score)

    #分别预估属于某个类的概率，注意其参数是一个[]列表
    print(knn.predict_proba([x_test[0],x_test[1],x_test[1000]]))


    for i in range(len(y_test_pred)):
    	if(y_test_pred[i]==y_test[i]):
    		prob+=1
    	else:
    		continue
    print("预测率:",(prob/len(y_test)))
