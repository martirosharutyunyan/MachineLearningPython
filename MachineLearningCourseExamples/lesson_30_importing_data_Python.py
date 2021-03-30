import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import mode


df = pd.read_csv('C:/Users/Martiros/Desktop/python/MachineLearningPython/pandasLibrary/House_Price.csv',header=0)
# print(df.head())
# print(df.shape)
# print(df.describe())
# sns.jointplot(data=df,x='price',y='crime_rate',kind='kde')
# sns.displot(data=df,kde=True,height=10)
# sns.jointplot(x='n_hot_rooms',y='price',data=df)
# sns.jointplot(x='rainfall',y='price',data=df)
# sns.countplot(x='airport',data=df)
# sns.countplot(x='waterbody',data=df)
# sns.countplot(x='bus_ter',data=df)
# print(df.info())
# print(np.percentile(df.n_hot_rooms,[99])[0])

# lesson 36
# n_hot_rooms data corecting
uv = np.percentile(df.n_hot_rooms,[99][0])
# print(df[(df.n_hot_rooms>uv)])
# print(df.n_hot_rooms[df.n_hot_rooms > 3*uv])
df.n_hot_rooms[(df.n_hot_rooms > 3*uv)] = 3*uv # minimization of df.n_hot_rooms
# sns.jointplot(x='n_hot_rooms',y='price',data=df)
# rainfalls data corecting
# print(np.percentile(df.rainfall,[1])[0])
lv = np.percentile(df.rainfall,[1][0])
# print(df.rainfall[df.rainfall < 0.3*lv])
# sns.jointplot(x='rainfall',y='price',data=df)

# print(df.info())
df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())
df.n_hos_beds = df.n_hos_beds.fillna(mode(df.n_hos_beds)[0][0]) # we filled the mising values with the mode value of df.n_hos_beds
# print(df.info())

# sns.regplot(data=df,y='price',x='rainfall')
# sns.jointplot(data=df,x='crime_rate',y='price')
df.crime_rate = np.log(1+df.crime_rate)
# sns.jointplot(data=df,x='crime_rate',y='price')
# sns.regplot(data=df,y='price',x='crime_rate')
df['avg_list'] = (df.dist1+df.dist2+df.dist3+df.dist4) / 4
# print(df.describe())
del df['dist1']
del df['dist2']
del df['dist3']
del df['dist4']
del df['bus_ter']
# print(df.describe())
# print(df.head())

# lesson 47 creating dummy variables
# print(df.head())
df = pd.get_dummies(df)
# print(df.head())
del df['airport_NO']
del df['waterbody_None']
del df['parks']
# print(df.head())

# sns.regplot(data=df,x='n_hot_rooms',y='price')
# y = df['price'].values
# x = df['n_hot_rooms'].values
# r = np.corrcoef(x=x,y=y)[0,1]
# print(r)
# a = np.correlate(x,y,mode='valid')
# print(a)

# lesson 50
# print(df.corr())
# del df['parks']

# Linear Regresion
# from sklearn.linear_model import LinearRegression
# y = df['price'].values.reshape(-1,1)
# x = df['room_num'].values.reshape(-1,1)
# reg = LinearRegression().fit(x,y)
# print(reg.coef_,reg.intercept_)
# print(reg.score(x,y))

print(df)

# filtering our data's n_hot_rooms
# data = np.array(df)
# print(data.shape)
# filtered_data = data[data[:,14] < 50]
# print(filtered_data.shape)
# filtered_data = pd.DataFrame(filtered_data,columns=df.columns)
# print(filtered_data)
# sns.jointplot(x='n_hot_rooms',y='price',data=filtered_data)
