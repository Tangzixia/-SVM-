#coding=utf-8

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree,metrics
import pydotplus
from IPython.display import Image,display
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor,bagging,weight_boosting
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score #交叉验证
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import AdaBoostingRegressor
from sklearn.ensemble import BaggingRegressor,ExtraTreesRegressor
from sklearn.cross_validation import KFold,StratifiedKFold,LeaveOneOut,LeavePOut,LeaveOneLabelOut,LeavePLabelOut


#在分类器或者回归器中总有random_state这个参数，那么这个参数是用来干嘛的呢？
#很多模型都需要一个随机的设定(比如迭代的初始值等等),random_state的作用就是固定这个随机设定
#调参的时候后，这个random_state通常是固定不变的
#这个可以看一个栗子，random_state和random.seed()其实是一样额
#random.seed(123)
#for i in range(10): print(random.randint(1,100))
#random.seed(123)
#for i in range(10): print(random.randint(1,100))
#
#我们运行这个程序两遍，可以发现程序的结果是一样的
#即用random_state参数，我们可以固定我们想要固定的随机数

housing=fetch_california_housing()
#print(housing.DESCR)
#获取到该数据集中的不同特征的名称
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
#print(housing.feature_names)
#print(housing.data.shape)

#观察对应的标签，我们可以发现
#对应的标签为连续型的，因此我们需要训练一个回归器进行处理
#[ 4.526  3.585  3.521 ...,  0.923  0.847  0.894]
#print(housing.target)

dtr=tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(housing.data[:,:],housing.target)

dot_data=tree.export_graphviz(dtr,out_file=None,feature_names=housing.feature_names[:],filled=True,impurity=False,rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("cal_house_all_1.pdf")
img=Image(graph.create_png())
#print(img.data)
#display(img)
#graph.write_png("dtr_white_background.png")
#cv=cross_validation.ShuffleSplit(n_samples,n_iter=3,test_size=0.3,random_state=0)


#决策树进行预测
#0.624977554157
data_train,data_test,target_train,target_test=\
		train_test_split(housing.data,housing.target,test_size=0.1,random_state=42)
dtr=tree.DecisionTreeRegressor()
dtr.fit(data_train,target_train)
#score=dtr.score(data_test,target_test)
kf=KFold(data_train.shape[0],n_folds=5)
skf=StratifiedKFold(target_train,n_folds=5)
loo=LeaveOneOut(data_train.shape[0])
lpo=LeavePOut(data_train.shape[0],6)

labels_lolo=[1,1,2,2]
lolo=LeaveOneLabelOut(labels_lolo)
#这个策略在划分样本时，会根据第三方提供的整数型样本类标号进行划分
#每次划分数据集时，去除某个属于某个类裱好的样本作为测试集，剩余的作为训练集
labels_lopo=[1,1,2,2,3,3]
lopo=LeavePLabelOut(labels_lopo,2)
#这个策略每次取得p种类标号的数据作为测试集，其余作为训练集

#注意cross_val_score中的cv参数
cross_score=cross_val_score(dtr,data_train,target_train,cv=skf)
print("交叉验证：")
print(cross_score)
#print(score)

'''
#这种无法使用metrics进行判断，原因在于continous is not supported
pred_test=dtr.predict(data_test)
acc_score=metrics.accuracy_score(target_test,pred_test)
print(acc_score)
'''

#随机森林进行回归预测
#0.79086492281
rfr=RandomForestRegressor(random_state=42)
rfr.fit(data_train,target_train)
score=rfr.score(data_test,target_test)
print(score)

'''
#svm.SVR()进行处理
#利用SVR()进行回归分析
#0.110271746962
svr=SVR()
svr.fit(data_train,target_train)
acc_score01=svr.score(data_test,target_test)
print(acc_score01)
'''
#梯度boostingRegressor
#0.779394011734
gbr=GradientBoostingRegressor()
gbr.fit(data_train,target_train)
acc_score02=gbr.score(data_test,target_test)
print(acc_score02)

#BaggingRegressor
#0.787106370939
br=BaggingRegressor()
br.fit(data_train,target_train)
acc_score03=br.score(data_test,target_test)
print(acc_score03)


#ExtraTreesRegressor
#0.792765227185
etr=ExtraTreesRegressor()
etr.fit(data_train,target_train)
acc_score04=etr.score(data_test,target_test)
print(acc_score04)
