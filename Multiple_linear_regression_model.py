import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\20th-august - mlr\20th - mlr\MLR\Investment.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]


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


#x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x=np.append(arr=np.full((50,1),42467).astype(int),values=x,axis=1)


import statsmodels.api as sm
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()



x_opt=x[:,[0,1,2,3,4]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


x_opt=x[:,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


bias=regressor.score(x_train,y_train)
bias

variance=regressor.score(x_test,y_test)
variance


plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.scatter(x=bias,y=variance)

import pickle
filename='multiple_linear_regression_model.pkl'
with open(filename,"wb") as file :
    pickle.dump(regressor,file)
print("Model has been pickled and saved as multiple_linear_regression_model.pkl")    

import os
print(os.getcwd())


















