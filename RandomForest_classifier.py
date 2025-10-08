import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\3-september-2.LOGISTIC REGRESSION CODE\2.LOGISTIC REGRESSION CODE\logit classification.csv")

x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(max_depth=4,n_estimators=60,random_state=0,criterion='entropy')
classifier.fit(x_train,y_train)


y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac

bias=classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance


dataset1=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\3-september-2.LOGISTIC REGRESSION CODE\2.LOGISTIC REGRESSION CODE\final1.csv")
d2=dataset1.copy()

dataset1=dataset1.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)
M


y_pred1=pd.DataFrame()
d2['y_pred1']=classifier.predict(M)
d2['y_pred1']



d2.to_csv('future_RandomForest_classifier.csv')

import os
os.getcwd()

























