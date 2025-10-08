import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\28-august-25th, 26th - Svr, Dtr, Rf, Knn\25th, 26th - Svr, Dtr, Rf, Knn\emp_sal.csv")


x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]


from sklearn.svm import SVR
svr_reg=SVR(degree=5,kernel="poly",gamma='auto')
svr_reg.fit(x,y)


svr_reg_pred=svr_reg.predict([[6.5]])
svr_reg_pred


#plt.scatter(x,y,color='red')
#plt.plot(svr_reg_pred,x,y,color='blue')
#plt.title('SVR Prediction')
#plt.xlabel('svr')
#plt.ylabel('Salary')
#plt.show()


from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=4)
knn_reg.fit(x,y)


knn_reg_pred=knn_reg.predict([[6.5]])
knn_reg_pred


from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor()
dt_reg.fit(x,y)


dt_reg_pred=dt_reg.predict([[6.5]])
dt_reg_pred


from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor()
rf_reg.fit(x,y)


rf_reg_pred=rf_reg.predict([[6.5]])
rf_reg_pred


#import xgboost as xg
#xgb_r=xg.XGBRegressor(objective='reg:linear',n_estimators=4)
#xgb_r.fit(x,y)

import xgboost as xg
xgb_r=xg.XGBRegressor()
xgb_r.fit(x,y)
xgb_reg_pred=xgb_r.predict([[6.5]])
xgb_reg_pred











