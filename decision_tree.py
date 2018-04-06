#coding=utf-8

from sklearn import tree,metrics
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
import graphviz as gv
from sklearn.model_selection import cross_val_score
x=[[0,0],[1,1],[3,3],[5,5,],[0,1],[2,1],[3,2]]
y=[0,1,0,1,0,0,1]

'''
clf=tree.DecisionTreeClassifier()
clf.fit(x,y)
print(clf.predict([[2,2]]))
print(clf.predict_proba([[2,2]]))
'''

iris=load_iris()
#根据criterion的不同取值确定是id3,c4.5还是cart算法
clf=tree.DecisionTreeClassifier(criterion="gini")
#clf.fit(iris.data,iris.target)

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
acc_score=metrics.accuracy_score(y_test,pred)
print(acc_score)
#acc=cross_val_score(clf,iris.data,iris.target,cv=10)
#print(acc)

dot_data=tree.export_graphviz(clf,out_file=None)
graph=gv.Source(dot_data)
#print(graph)


clf1=tree.DecisionTreeClassifier(criterion="entropy")
clf1.fit(x_train,y_train)
pred1=clf1.predict(x_test)
acc_score1=metrics.accuracy_score(y_test,pred1)
print(acc_score1)
#print(clf.predict_proba(iris.data[:1,:]))
#print(clf.predict(iris.data[:200]))
#graph.render("iris")
