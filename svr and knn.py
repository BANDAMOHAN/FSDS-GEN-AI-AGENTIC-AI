import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\25-august-poly\emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


from sklearn.svm import SVR
svr_regressor=SVR(kernel='poly',degree=4,gamma='auto',C=10.0)
svr_regressor.fit(x,y)


svr_model_pred=svr_regressor.predict([[6.5]])
print(svr_model_pred)

plt.scatter(x,y,color='red')
plt.plot(x,svr_regressor.predict(x),color='blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


from sklearn.neighbors import KNeighborsRegressor
knn_reg_model=KNeighborsRegressor(n_neighbors=3,weights='uniform')
knn_reg_model.fit(x,y)

knn_reg_pred=knn_reg_model.predict([[6.5]])
print(knn_reg_pred)






