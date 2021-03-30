import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression


x = np.arange(1,50)
y = x**2
data = pd.DataFrame([],columns=['x','y'])
data['x'], data['y'] = x, y
sns.regplot(data=data,x='x',y='y')
x = x.reshape(-1,1)
y = y.reshape(-1,1)
reg = LinearRegression().fit(x,y)
print(reg.score(x,y))
print(reg.coef_,reg.intercept_,reg.singular_)