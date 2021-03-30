import seaborn as sns
import pandas as pd
from json import loads
import numpy as np
from numba import jit, njit, prange
from scipy import stats

data1 = pd.read_csv('C:/Users/Martiros/Desktop/ /python/MachineLearningPython/pandasLibrary/Customer.csv',header=0)
# Seaborn lesson 1
# sns.displot(data1.Age,kde=True)
tips = sns.load_dataset('tips')
# print(data1.head())
# print(tips)
# print(tips.shape)
# print(data1.describe())
# print(stats.mode(data1)[0])
# sns.pairplot(tips)
# sns.jointplot(data=tips)
sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
# sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')
# sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
# sns.jointplot(x='total_bill',y='tip',data=tips,kind='kde')
# sns.rugplot(tips['total_bill'])
# sns.kdeplot(tips['total_bill'])
# sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std) 
# sns.boxplot(x='day',y='total_bill',hue='smoker',data=tips)  
# sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)
# sns.stripplot(data=tips,y='total_bill',x='day',hue='sex',jitter=True)
# sns.stripplot(data=tips,y='total_bill',x='day',hue='sex',jitter=True,split=True)
# sns.stripplot(data=tips,y='total_bill',x='day',hue='sex',split=True)
# sns.swarmplot(data=tips,y='total_bill',x='day',hue='sex')
# sns.factorplot(data=tips,y='total_bill',x='day',hue='sex')

# Seaborn lesson 2
# tips = sns.load_dataset('tips')
# fly = sns.load_dataset('flights')
# print(tips.head())
# print(fly.head())
# print(fly.describe()['passengers'])
# fly.groupby('month').sum()
# fly.groupby('month').std()
# fly.groupby('month').mean()
# fly.groupby('month')['passengers'].mean()
# fly.corr() # Correlyacia
# ts = tips.corr()
# fvt = fly.pivot_table(index='year',columns='month',values='passengers')
# sns.heatmap(data=fvt)
# sns.heatmap(data=fvt,linewidths=2)
# sns.clustermap(fvt,standard_scale=1)
# sns.clustermap(fvt,standard_scale=100)
















