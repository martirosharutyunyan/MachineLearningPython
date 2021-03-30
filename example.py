from sklearn.cluster import KMeans
from pandas import DataFrame
import seaborn as sns
import numpy as np
from numpy import exp
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import zscore, chi2_contingency
df = pd.read_csv(
    'C:/Users/Martiros/Desktop/python/MachineLearningPython/pandasLibrary/House_Price.csv', header=0)
y = df['price']
x = df[['crime_rate']]
reg = LinearRegression().fit(x, y)
# print(reg.coef_,reg.intercept_)


# z բաշխում
# df = pd.get_dummies(df)
# df_zscore = zscore(df)
# df = pd.DataFrame(df_zscore,columns=df.columns)
# print(df)
# https://static-v.tawk.to/709/app.js
# observed = ([10,6],[5,15])
# print(chi2_contingency(observed))

# x = np.arange(-10,10)
# y = x**2-4*x+3
# df = pd.DataFrame(columns=['X','Y'])
# df['X'],df['Y'] = x,y
# sns.jointplot(data=df,x='X',y='Y')

# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression().fit(X,y)
# print(clf.score(X,y))


Data = {
    'x': [-3, 1, 2, 3, 5, 6, 7],
    'y': [3, 4, 6, 8, 2, 11, 1]
}

df = DataFrame(Data, columns=['x', 'y'])

kmeans = KMeans(1)
kmeans.fit(df)
print(kmeans.inertia_)
