#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets,linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
from sklearn.model_selection import cross_val_predict


data=pd.read_csv(".\CCPP\ccpp.csv")

x=data[['AT','V','AP','RH']]
y=data['PE']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

linereg=LinearRegression()
linereg.fit(x_train, y_train)
print(linereg.intercept_)
print(linereg.coef_)


y_pred=linereg.predict(x_test)
#MSE: 20.7774781069
#RMSE: 4.55823190578
print("MSE:",metrics.mean_squared_error(y_test,y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

X=data[['AT','V','AP','RH']]
Y=data[['PE']]

#通过交叉验证来优化模型，可以发现结果为：
#MSE: 20.7936725099
#RMSE: 4.56000795064
#注意这儿MSE或者RMSE均比前面计算出来的值更高，不要怀疑交叉验证的效果
#原因在于我们这儿将所有这的样本做测试集，求其对应预测值得MSE
#而前面仅仅只计算了少量测试集上的MSE
predicted=cross_val_predict(linereg,X,Y,cv=10)
print("MSE:",metrics.mean_squared_error(Y,predicted))
print("RMSE:",np.sqrt(metrics.mean_squared_error(Y,predicted)))

#画图观察结果
fig,ax=plt.subplots()
ax.scatter(y,predicted)
ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()
