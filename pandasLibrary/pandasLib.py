import pandas as pd
from json import loads
import json
import numpy as np

data1 = pd.read_csv('C:/Users/Martiros/Desktop/MachineLearningPython/pandasLibrary/Customer.csv',header=0)
data2 = pd.read_csv('C:/Users/Martiros/Desktop/MachineLearningPython/pandasLibrary/Customer.csv',header=0,index_col=0)
# print(data2.head(10))
arr = np.array(data2.head()['Age'])
print(arr)
print(arr.std())
print(arr.mean())
# print(data1.describe())
# print(data1.iloc[0])
# print(data2.loc['CG-12520'])
# print(data2.iloc[0:5])