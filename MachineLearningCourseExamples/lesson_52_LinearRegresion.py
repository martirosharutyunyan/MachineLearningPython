import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sn
from lesson_30_importing_data_Python import df
from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy.stats as stats



# URL = 'https://stepik.org/media/attachments/lesson/8083/genetherapy.csv'
# data = pd.read_csv(URL)
# A = data[data["Therapy"] == "A"]["expr"]
# B = data[data["Therapy"] == "B"]["expr"]
# C = data[data["Therapy"] == "C"]["expr"]
# D = data[data["Therapy"] == "D"]["expr"]
# print(stats.f_oneway(A, B, C, D))
print(df.corr())


# tarberak 1 
# X = sn.add_constant(df['room_num'])
# lm = sn.OLS(df['price'],X).fit()

# tarberak 2
# y = df['price']
# X = df[['room_num']]
# lm2 = LinearRegression().fit(X,y)
# print(lm2.intercept_,lm2.coef_)
# sns.jointplot(data=df,x='room_num',y='price',kind='reg')

# im tarberaky
# from scipy.stats import sem
# sns.regplot(data=df,x='room_num',y='price')
# X = df['room_num'].values.reshape(-1,1)
# y = df['price'].values.reshape(-1,1)
# model = LinearRegression().fit(X,y)
# print(model.intercept_,model.coef_)
# print(sem(df['room_num'].values))
# print(model.score(X,y))

# lesson 61 multiple linear regression
X_multi = df.drop('price',axis=1)
y_multi = df['price']
# X_muti_cons = sn.add_constant(X_multi)
# lm_multi = sn.OLS(y_multi,X_muti_cons).fit()
# print(lm_multi.summary())
# lm3 = LinearRegression().fit(X_multi,y_multi)
# print(lm3.intercept_,lm3.coef_)

# lesson 65 
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# X_train, X_test, y_train, y_test = train_test_split(X_multi,y_multi,test_size=0.2,random_state=0)
# lm_a = LinearRegression().fit(X_train,y_train)
# y_test_a = lm_a.predict(X_test)
# y_train_a = lm_a.predict(X_train)
# print(r2_score(y_test,y_test_a)) 
# print(r2_score(y_train,y_train_a))


# dispersia
# x = np.array([1,2,3,4,5]).reshape(-1,1)
# print(x.std())
# print(x.mean())
# new_data_frame = pd.DataFrame(data=x,columns=['X'])
# print(new_data_frame.describe())