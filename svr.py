import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv(r"D:\Data Science with AI\Data Science With AI\25-august-poly\emp_sal.csv")

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

from sklearn.svm import SVR
svr_regressor=SVR()
svr_regressor.fit(x,y)


svr_model_pred