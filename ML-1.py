import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\13th-august- ML\13th- ML\5. Data preprocessing\Data.csv")
x=dataset.iloc[:,:-1].values
z=dataset.iloc[:,:3]
z2=dataset.iloc[:,:-2]
z3=dataset.iloc[:,-2]
z1=dataset.iloc[:,-1]
z4=dataset.iloc[:,-4]
z5=dataset.iloc[:-2]
x2=dataset.iloc[:,:-1]
x3=dataset.iloc[:,3]
x4=dataset.iloc[:,:-2]
x5=dataset.iloc[:,2]
x6=dataset.iloc[:,:-3]
x7=dataset.iloc[:,1]
x8=dataset.iloc[:,:-4]
x9=dataset.iloc[:,0]
x10=dataset.iloc[:,2:3]
x11=dataset.iloc[:,:3]
x12=dataset.iloc[:,:2]
x13=dataset.iloc[:,:0]
x14=dataset.iloc[:,1:3]
y=dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer()
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])



from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0]=labelencoder_x.fit_transform(x[:,0])


labelencoder_y=LabelEncoder()
labelencoder_y.fit_transform(y)
y=labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
