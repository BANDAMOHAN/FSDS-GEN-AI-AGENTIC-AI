import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\19th-august- slr with streamlit\18th- slr with streamlit\Salary_Data.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


regressor=LinearRegression()
regressor.fit(x_train,y_train)


y_pred=regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')

plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_12=regressor.predict([[12]])
y_20=regressor.predict([[20]])

print(f"Predicted salary for 12 years of experience:${y_12[0]:,.2f}")
print(f"Predicted salary for 20 years of experience:${y_20[0]:,.2f}")


bias=regressor.score(x_train,y_train)
variance=regressor.score(x_test,y_test)
train_mse=mean_squared_error(y_train,regressor.predict(x_train))
test_mse=mean_squared_error(y_test,y_pred)


print(f"Training Score(R^2):{bias:.2f}")
print(f"Testing Score(R^2):{variance:.2f}")
print(f"Training MSE:{train_mse:.2f}")
print(f"Test MSE:{test_mse:.2f}")

filename='linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pkl")


import os
print(os.getcwd())




















