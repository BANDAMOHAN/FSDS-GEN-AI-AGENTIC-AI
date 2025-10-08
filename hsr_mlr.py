import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\20th-august - mlr\20th - mlr\MLR\House_data.csv")

x=dataset.iloc[:,3:]
y=dataset.iloc[:,2]


x=pd.get_dummies(x,dtype=int)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)


m=regressor.coef_
print(m)


c=regressor.intercept_
print(c)


x=np.append(arr=np.full((21613,1),4267).astype(int),values=x,axis=1)

import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

import pickle
filename='hsp_mlr.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print('Model has been pickled')

import os
os.get













