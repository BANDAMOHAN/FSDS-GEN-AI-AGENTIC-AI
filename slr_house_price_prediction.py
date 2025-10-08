import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\15th-august- SLR\15th- SLR\SLR - House price prediction\House_data.csv")

space=dataset['sqft_living']
price=dataset['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)


pred=regressor.predict(xtest)


plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Visuals for Training Dataset')
plt.xlabel('space')
plt.ylabel('price')
plt.show()


plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title('Visuals for Test Dataset')
plt.xlabel('space')
plt.ylabel('Price')
plt.show()



import pickle
filename='House_price_prediction_slr.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print('Model has been pickled and saved as House_price_prediction_slr.pkl')
















