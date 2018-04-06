#coding=utf-8

from sklearn import svm,metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv("./CCPP/bmi.csv")


label=data["label"]

y=np.zeros(label.shape)
y[label=="normal"]=1
y[label=="fat"]=2
fea_h=data["height"]/100
fea_w=data["weight"]/100

fea_all=pd.concat([fea_w,fea_h],axis=1)
x_train,x_test,y_train,y_test=train_test_split(fea_all,y,test_size=0.2)

svm_=svm.SVC(decision_function_shape="ovo")
svm_.fit(x_train,y_train)
dec=svm_.decision_function([[1,1]])
print(dec.shape)

predict=svm_.predict(x_test)
ac_score=metrics.accuracy_score(y_test,predict)
cl_report=metrics.classification_report(y_test,predict)

print(ac_score)

svm_01=svm.SVC(decision_function_shape="ovr")
svm_01.fit(x_train,y_train)
dec_01=svm_01.decision_function([[1,1]])
print(dec_01.shape)
predict_01=svm_01.predict(x_test)
ac_score_01=metrics.accuracy_score(y_test,predict_01)
print(ac_score_01)


svm_02=svm.SVC(kernel="rbf")
svm_02.fit(x_train,y_train)
dec_02=svm_02.decision_function([[1,1]])
print(dec_02.shape)
predict_02=svm_02.predict(x_test)
ac_score_02=metrics.accuracy_score(y_test,predict_02)
print(ac_score_02)

svm_03=svm.SVC(kernel="poly")
svm_03.fit(x_train,y_train)
dec_03=svm_03.decision_function([[1,1]])
print(dec_03.shape)
predict_03=svm_03.predict(x_test)
ac_score_03=metrics.accuracy_score(y_test,predict_03)
print(ac_score_03)

svm_04=svm.SVC(kernel="linear")
svm_04.fit(x_train,y_train)
dec_04=svm_04.decision_function([[1,1]])
print(dec_04.shape)
predict_04=svm_04.predict(x_test)
ac_score_04=metrics.accuracy_score(y_test,predict_04)
print(ac_score_04)



#这下我们采用了LinearSVC，可以观察结果发现LinearSVC的结果和SVC(decision_function_shape="ovr")的结果并不一致
#造成这种结果的原因在于LinearSVC和SVC(默认是"ovo")的
#1)LinearSVC的decision_function_shape="ovr"
#2)LinearSVC的损失函数是squared Hinge Loss，而SVC的损失函数是Hinge Loss
svm__=svm.LinearSVC()
svm__.fit(x_train,y_train)
dec_=svm__.decision_function([[1,1]])
print(dec_.shape)
predict_=svm__.predict(x_test)
ac_score_=metrics.accuracy_score(y_test,predict_)
print(ac_score_)
#print(cl_report)
